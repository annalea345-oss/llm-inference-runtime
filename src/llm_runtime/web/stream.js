// Small SSE decoder shared by the browser client and dependency-free Node tests.
export class ChatStreamError extends Error {
  constructor(message, code = 'invalid_stream') { super(message); this.name = 'ChatStreamError'; this.code = code; }
}

export class SSEDecoder {
  constructor() { this.buffer = ''; this.data = []; this.size = 0; }
  feed(text, final = false) {
    this.buffer += text;
    const events = [];
    while (true) {
      const index = this.buffer.search(/[\r\n]/);
      if (index < 0 || (!final && index === this.buffer.length - 1 && this.buffer[index] === '\r')) break;
      const line = this.buffer.slice(0, index);
      const width = this.buffer[index] === '\r' && this.buffer[index + 1] === '\n' ? 2 : 1;
      this.buffer = this.buffer.slice(index + width);
      if (line === '') {
        if (this.data.length) events.push(this.data.join('\n'));
        this.data = []; this.size = 0;
      } else if (line === 'data' || line.startsWith('data:')) {
        const value = line === 'data' ? '' : line.slice(5).replace(/^ /, '');
        this.data.push(value); this.size += value.length;
      }
      if (this.size > 1048576) throw new ChatStreamError('The server sent an oversized event.');
    }
    if (this.buffer.length + this.size > 1048576) throw new ChatStreamError('The server sent an oversized event.');
    // An event without its closing blank line is incomplete; do not dispatch it.
    return events;
  }
}

async function* dataEvents(response) {
  if (!response.body) throw new ChatStreamError('This response has no readable stream.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8', {fatal: true});
  const sse = new SSEDecoder();
  try {
    while (true) {
      const {value, done} = await reader.read();
      for (const event of sse.feed(done ? decoder.decode() : decoder.decode(value, {stream: true}), done)) yield event;
      if (done) return;
    }
  } finally {
    try { await reader.cancel(); } catch { /* The connection may already be closed. */ }
    reader.releaseLock();
  }
}

export async function consumeChatStream(response, {onText = () => {}, model} = {}) {
  let identity = null, finish = null, usage = null;
  for await (const event of dataEvents(response)) {
    if (event === '[DONE]') {
      if (!finish || !usage) throw new ChatStreamError('The stream ended without completion details.');
      return {finish, usage, id: identity.id};
    }
    let chunk;
    try { chunk = JSON.parse(event); } catch { throw new ChatStreamError('The server sent invalid stream data.'); }
    if (!chunk || typeof chunk !== 'object' || Array.isArray(chunk)) throw new ChatStreamError('The server sent an unexpected response format.');
    if (chunk.error) throw new ChatStreamError(chunk.error.message || 'The request failed.', chunk.error.code || 'server_error');
    if (chunk.object !== 'chat.completion.chunk' || typeof chunk.id !== 'string' ||
        !Number.isInteger(chunk.created) || typeof chunk.model !== 'string' || !Array.isArray(chunk.choices)) {
      throw new ChatStreamError('The server sent an unexpected response format.');
    }
    identity ??= {id: chunk.id, created: chunk.created, model: chunk.model};
    if (chunk.id !== identity.id || chunk.created !== identity.created || chunk.model !== identity.model ||
        (model && chunk.model !== model)) throw new ChatStreamError('The stream changed request identity.');
    if (chunk.choices.length === 0) {
      const u = chunk.usage;
      if (!finish || usage || !u || ![u.prompt_tokens, u.completion_tokens, u.total_tokens].every(v => Number.isInteger(v) && v >= 0) ||
          u.total_tokens !== u.prompt_tokens + u.completion_tokens) throw new ChatStreamError('The stream has invalid token usage.');
      usage = u; continue;
    }
    const choice = chunk.choices[0];
    if (usage || finish || chunk.choices.length !== 1 || choice?.index !== 0 || !choice.delta ||
        typeof choice.delta !== 'object' || Array.isArray(choice.delta) || (choice.delta.role && choice.delta.role !== 'assistant')) {
      throw new ChatStreamError('The stream has an invalid response sequence.');
    }
    const text = choice.delta.content ?? '';
    if (typeof text !== 'string') throw new ChatStreamError('The stream has invalid text.');
    if (text) onText(text);
    if (choice.finish_reason != null) {
      if (!['stop', 'length'].includes(choice.finish_reason)) throw new ChatStreamError('Unknown completion reason.');
      finish = choice.finish_reason;
    }
  }
  throw new ChatStreamError('The connection closed before the answer finished.');
}
