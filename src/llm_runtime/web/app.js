import {consumeChatStream, ChatStreamError} from './stream.js';

const $ = id => document.getElementById(id);
const turns = [];
let active = null, online = false, modelsLoaded = false, loadingModels = null;
let lastSnapshot = null;
const number = value => new Intl.NumberFormat().format(value);
const duration = ms => ms == null ? '—' : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
const bytes = n => n >= 1048576 ? `${(n / 1048576).toFixed(2)} MiB` : n >= 1024 ? `${(n / 1024).toFixed(1)} KiB` : `${n} B`;

function notice(text = '') { $('notice').textContent = text; $('notice').hidden = !text; }
function controls() {
  $('send').disabled = Boolean(active) || !online || !modelsLoaded;
  $('send').hidden = Boolean(active); $('stop').hidden = !active;
  $('stop').disabled = Boolean(active?.stopRequested);
  $('stop').firstChild.textContent = active?.stopRequested ? 'Stopping ' : 'Stop ';
  for (const id of ['new-chat', 'model', 'system-prompt', 'max-tokens']) $(id).disabled = Boolean(active) || (id === 'model' && !modelsLoaded);
}
function node(tag, className, text) {
  const element = document.createElement(tag); element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function addTurn(user) {
  $('welcome').hidden = true;
  const container = node('article', 'turn');
  const userElement = node('div', 'message user');
  const userLabel = node('div', 'message-label', 'YOU');
  userElement.append(userLabel, node('div', 'message-text', user));
  const assistant = node('div', 'message assistant pending');
  const label = node('div', 'message-label'); label.append(node('span', 'mini-dot'), document.createTextNode('ASSISTANT'));
  const text = node('div', 'message-text');
  const footer = node('div', 'message-footer', 'Waiting for output…');
  assistant.append(label, text, footer); container.append(userElement, assistant);
  $('conversation').append(container);
  const turn = {user, text: '', status: 'pending', element: container, assistant, output: text, footer};
  turns.push(turn); scrollDown(true); return turn;
}
function scrollDown(force = false) {
  const view = $('conversation');
  if (force || view.scrollHeight - view.scrollTop - view.clientHeight < 150) view.scrollTop = view.scrollHeight;
}
function conversationMessages(user) {
  const messages = [];
  if ($('system-prompt').value.trim()) messages.push({role: 'system', content: $('system-prompt').value});
  for (const turn of turns) {
    if (turn.status === 'completed' && turn.text) messages.push({role: 'user', content: turn.user}, {role: 'assistant', content: turn.text});
  }
  messages.push({role: 'user', content: user});
  if (messages.length > 64 || messages.reduce((sum, m) => sum + Array.from(m.content).length, 0) > 8192) {
    throw new Error('This conversation exceeds the input limit. Start a new chat or shorten the instructions.');
  }
  return messages;
}
async function responseError(response) {
  try { const body = await response.json(); return body.error?.message || body.detail || `Request failed (${response.status}).`; }
  catch { return `Request failed (${response.status}).`; }
}

$('chat-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (active || !online || !modelsLoaded) return;
  const prompt = $('prompt').value;
  if (!prompt.trim()) { $('prompt').focus(); return; }
  let messages;
  const maxTokens = Number($('max-tokens').value);
  try {
    messages = conversationMessages(prompt);
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) throw new Error('Choose an output limit from 1 to 512 tokens.');
  } catch (error) { notice(error.message); return; }
  notice();
  const turn = addTurn(prompt);
  const state = {controller: new AbortController(), requestId: null, stopRequested: false, started: performance.now(), first: null};
  active = state; controls();
  $('prompt').value = ''; $('input-count').textContent = '0 characters';
  $('first-text').textContent = $('total-time').textContent = $('output-tokens').textContent = '—';
  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, signal: state.controller.signal,
      body: JSON.stringify({model: $('model').value, messages, max_completion_tokens: maxTokens,
        temperature: 0, stream: true, stream_options: {include_usage: true}}),
    });
    if (!response.ok) throw new ChatStreamError(await responseError(response), 'http_error');
    state.requestId = response.headers.get('X-Request-ID');
    const result = await consumeChatStream(response, {model: $('model').value, onText(text) {
      if (state.first === null) {
        state.first = performance.now() - state.started; $('first-text').textContent = duration(state.first);
      }
      turn.text += text; turn.output.textContent = turn.text; turn.footer.textContent = 'Receiving answer…'; scrollDown();
    }});
    turn.status = 'completed';
    turn.footer.textContent = `${number(result.usage.completion_tokens)} tokens · ${duration(performance.now() - state.started)} · ${result.finish === 'length' ? 'Output limit reached' : 'Finished'}`;
    $('output-tokens').textContent = number(result.usage.completion_tokens);
    if (!turn.text) turn.output.textContent = 'No visible text returned.';
  } catch (error) {
    const cancelled = error.name === 'AbortError' || error.code === 'request_cancelled';
    turn.status = cancelled ? 'cancelled' : 'failed';
    turn.footer.textContent = cancelled ? 'Stopped · excluded from the next conversation context' : 'Failed · excluded from the next conversation context';
    if (!turn.text) turn.output.textContent = cancelled ? 'Generation stopped.' : 'No answer received.';
    if (!cancelled) notice(error.message || 'Unable to reach the runtime.');
  } finally {
    turn.assistant.classList.remove('pending');
    $('total-time').textContent = duration(performance.now() - state.started);
    if (active === state) active = null;
    controls(); $('prompt').focus();
  }
});

$('stop').addEventListener('click', async () => {
  const state = active;
  if (!state || state.stopRequested) return;
  state.stopRequested = true; controls();
  if (!state.requestId) { state.controller.abort(); return; }
  try {
    const response = await fetch(`/requests/${encodeURIComponent(state.requestId)}/cancel`, {method: 'POST', signal: AbortSignal.timeout(5000)});
    // A 404 can mean completion won the race; let the answer stream settle.
    if (!response.ok && response.status !== 404) throw new Error(await responseError(response));
  } catch {
    state.controller.abort();
    if (active === state) notice('The cancellation endpoint was unavailable. The stream was closed; cleanup is cooperative.');
  }
});
$('new-chat').addEventListener('click', () => {
  if (active) return;
  for (const turn of turns) turn.element.remove();
  turns.length = 0; $('welcome').hidden = false; notice();
  $('first-text').textContent = $('total-time').textContent = $('output-tokens').textContent = '—';
  $('prompt').value = ''; $('input-count').textContent = '0 characters'; $('prompt').focus();
});
$('prompt').addEventListener('input', () => { $('input-count').textContent = `${number(Array.from($('prompt').value).length)} characters`; });
$('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('chat-form').requestSubmit(); }
});
for (const button of document.querySelectorAll('[data-prompt]')) button.addEventListener('click', () => {
  if (active) return;
  $('prompt').value = button.dataset.prompt; $('prompt').dispatchEvent(new Event('input')); $('prompt').focus();
});
window.addEventListener('pagehide', () => active?.controller.abort());

async function loadModels() {
  if (loadingModels) return loadingModels;
  loadingModels = (async () => {
    const response = await fetch('/v1/models', {signal: AbortSignal.timeout(3000), cache: 'no-store'});
    if (!response.ok) throw new Error('Model discovery failed.');
    const body = await response.json();
    if (!body.data?.length) throw new Error('No model is served.');
    $('model').replaceChildren(...body.data.map(model => new Option(model.id, model.id)));
    modelsLoaded = true;
  })();
  try { await loadingModels; } finally { loadingModels = null; }
}
function renderHealth(health) {
  $('waiting-count').textContent = number(health.queued_requests);
  $('active-count').textContent = number(health.active_requests);
  $('capacity-note').textContent = `${health.config.max_active_requests} active slots · ${health.config.queue_capacity} waiting slots`;
  const cache = health.kv_cache;
  $('capacity-total').textContent = `${number(cache.capacity_tokens)} slots`;
  $('used-value').textContent = number(cache.used_tokens); $('reserved-value').textContent = number(cache.reserved_tokens);
  $('available-value').textContent = number(cache.available_tokens);
  $('reserved-bar').style.width = `${100 * cache.reserved_tokens / cache.capacity_tokens}%`;
  $('used-bar').style.width = `${100 * cache.used_tokens / cache.capacity_tokens}%`;
  $('capacity-bar').setAttribute('aria-label', `${cache.used_tokens} used, ${cache.reserved_tokens} reserved of ${cache.capacity_tokens} token slots`);
  const real = health.executor.backend === 'transformers';
  const simulated = health.executor.backend === 'simulated';
  $('backend-label').textContent = real ? 'Local model · CPU' : simulated ? 'Simulation' : 'Custom executor';
  $('backend-label').classList.toggle('real', real);
  $('kv-bytes').textContent = Number.isFinite(health.executor.kv_tensor_bytes_after_last_step) ? bytes(health.executor.kv_tensor_bytes_after_last_step) : '—';
  $('memory-note').textContent = simulated ? 'Simulation has no real model tensors. Token slots above are accounting units.' : 'KV tensor snapshot after model steps. Weights and temporary calculations are separate.';
  $('policy-label').textContent = health.config.scheduling_policy === 'decode_first' ? 'Decode-first scheduling' : 'Round-robin scheduling';
  $('request-count').textContent = number(health.request_states.length);
  const rows = health.request_states.map(request => {
    const row = node('div', 'request-row'); const title = node('div', 'request-title');
    const phase = request.cancel_requested ? 'Stopping' : {waiting: 'Waiting', prefill: 'Reading prompt', decode: 'Answering'}[request.phase] || request.phase;
    title.append(node('code', '', request.request_id.slice(0, 8)), node('span', `request-phase ${request.phase}`, phase));
    const detail = node('div', 'request-detail');
    const progress = request.phase === 'prefill' ? `${request.prefilled_tokens}/${request.prompt_tokens} input tokens` : `${request.generated_tokens}/${request.max_tokens} output tokens`;
    detail.append(node('span', '', progress), node('span', '', active?.requestId === request.request_id ? 'This client' : 'Other request'));
    row.append(title, detail); return row;
  });
  $('request-list').replaceChildren(...(rows.length ? rows : [node('p', 'empty-state', 'No live requests. Capacity is ready.')]));
}
async function pollHealth() {
  try {
    const response = await fetch('/health', {signal: AbortSignal.timeout(3000), cache: 'no-store'});
    if (!response.ok) throw new Error('Health check failed.');
    const health = await response.json();
    const reconnecting = !online;
    online = health.status === 'ok';
    if (online && (!modelsLoaded || reconnecting)) await loadModels();
    renderHealth(health);
    lastSnapshot = Date.now(); $('snapshot-note').textContent = 'Updated just now · all clients';
    $('connection-status').textContent = online ? 'Runtime connected' : 'Runtime unavailable';
    $('live-indicator').textContent = online ? 'Live' : 'Unavailable';
  } catch {
    online = false; $('connection-status').textContent = 'Runtime disconnected'; $('live-indicator').textContent = 'Offline';
    $('snapshot-note').textContent = lastSnapshot === null ? 'Waiting for a runtime connection.' : `Last update ${Math.floor((Date.now() - lastSnapshot) / 1000)}s ago · values may be stale`;
  } finally {
    $('connection-dot').classList.toggle('online', online); controls();
    setTimeout(pollHealth, document.hidden ? 1500 : 500);
  }
}
pollHealth();
