import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source = readFileSync(new URL('../src/llm_runtime/web/stream.js', import.meta.url));
const {consumeChatStream, SSEDecoder, ChatStreamError} = await import('data:text/javascript;base64,' + source.toString('base64'));

const envelope = choices => ({id: 'chatcmpl-example', object: 'chat.completion.chunk', created: 1, model: 'test', choices});
const choice = (delta = {}, finish_reason = null) => ({index: 0, delta, finish_reason});
const frame = object => `data: ${typeof object === 'string' ? object : JSON.stringify(object)}\n\n`;
const usage = {prompt_tokens: 3, completion_tokens: 2, total_tokens: 5};
const valid = [envelope([choice({role: 'assistant', content: ''})]), envelope([choice({content: 'café'})]),
  envelope([choice({}, 'stop')]), {...envelope([]), usage}, '[DONE]'];
function response(events, split = 7) {
  const bytes = new TextEncoder().encode(events.map(frame).join(''));
  return new Response(new ReadableStream({start(controller) {
    for (let i = 0; i < bytes.length; i += split) controller.enqueue(bytes.slice(i, i + split));
    controller.close();
  }}));
}

test('split UTF-8 bytes and every boundary preserve content and final usage', async () => {
  for (const split of [1, 2, 7, 53, 4096]) {
    let text = '';
    const result = await consumeChatStream(response(valid, split), {model: 'test', onText: t => { text += t; }});
    assert.equal(text, 'café'); assert.deepEqual(result, {finish: 'stop', usage, id: 'chatcmpl-example'});
  }
});
test('CRLF split across calls, comments, and multiline data are assembled', () => {
  const parser = new SSEDecoder();
  assert.deepEqual(parser.feed(': ping\r'), []);
  assert.deepEqual(parser.feed('\ndata: {"a":\r\ndata: 1}\r'), []);
  assert.deepEqual(parser.feed('\n\r'), []);
  assert.deepEqual(parser.feed('\n'), ['{"a":\n1}']);
});
test('unterminated data is never presented as a completed event', () => {
  const parser = new SSEDecoder();
  assert.deepEqual(parser.feed('data: [DONE]', true), []);
});
test('failed stream is not a completed answer', async () => {
  await assert.rejects(consumeChatStream(response([{error: {message: 'stopped', code: 'request_cancelled'}}])),
    error => error instanceof ChatStreamError && error.code === 'request_cancelled');
});
for (const [name, events] of [
  ['missing DONE', valid.slice(0, -1)], ['missing finish', [valid[0], valid[1], valid[3], '[DONE]']],
  ['missing usage', [valid[0], valid[1], valid[2], '[DONE]']],
  ['changed identity', [valid[0], {...valid[1], id: 'other'}, ...valid.slice(2)]],
  ['wrong model', [valid[0], {...valid[1], model: 'other'}, ...valid.slice(2)]],
  ['content after finish', [valid[0], valid[2], valid[1], valid[3], '[DONE]']],
  ['invalid usage', [valid[0], valid[2], {...valid[3], usage: {...usage, total_tokens: 7}}, '[DONE]']],
  ['duplicate usage', [...valid.slice(0, -1), valid[3], '[DONE]']],
  ['primitive JSON', ['null']], ['malformed JSON', ['{broken']],
  ['null choice', [envelope([null])]], ['array delta', [envelope([choice([], 'stop')])]],
]) test(`rejects ${name}`, async () => { await assert.rejects(consumeChatStream(response(events)), ChatStreamError); });
test('reader cancellation propagates when the HTTP request is aborted', async () => {
  const body = new ReadableStream({pull(controller) { controller.error(new DOMException('aborted', 'AbortError')); }});
  await assert.rejects(consumeChatStream(new Response(body)), error => error.name === 'AbortError');
});
test('model output remains raw text for the rendering layer', async () => {
  const html = '<img src=x onerror="alert(1)">'; let text;
  const events = [valid[0], envelope([choice({content: html})]), ...valid.slice(2)];
  await consumeChatStream(response(events), {onText: value => { text = value; }});
  assert.equal(text, html);
});
