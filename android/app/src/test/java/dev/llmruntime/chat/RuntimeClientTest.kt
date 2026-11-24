package dev.llmruntime.chat

import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.*

class RuntimeClientTest {
    private fun server(test: (MockWebServer, String) -> Unit) {
        MockWebServer().use { s -> s.start(); test(s, s.url("/").toString().trimEnd('/')) }
    }
    private val direct = Executor { it.run() }
    private fun chunk(choices: String, usage: String = "null") =
        """data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,"model":"local-model","choices":$choices,"usage":$usage}""" + "\n\n"
    private val answer = chunk("""[{"index":0,"delta":{"content":"blåbær 🌍"},"finish_reason":null}]""")
    private val finish = chunk("""[{"index":0,"delta":{},"finish_reason":"stop"}]""")
    private val usage = chunk("[]", """{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}""")
    private fun response(body: String) = MockResponse().setHeader("X-Request-ID", "abc")
        .setHeader("Content-Type", "text/event-stream; charset=utf-8").setBody(body)

    @Test fun discoveryAndStreamingOverRealHttp() = server { s, base ->
        s.enqueue(MockResponse().setBody("""{"data":[{"id":"local-model"}]}"""))
        val body = answer + finish + usage + "data: [DONE]\n\n"
        s.enqueue(response(body).setChunkedBody(body, 1))
        assertEquals("local-model", discover(base))
        val text = StringBuilder()
        val result = Generation(base, direct).run("local-model", listOf(Message("user", "hello")), 32) { text.append(it) }
        assertEquals("blåbær 🌍", text.toString()); assertEquals(Completion("stop", 1), result)
        assertEquals("/v1/models", s.takeRequest().path)
        val request = s.takeRequest()
        assertEquals("POST", request.method)
        assertEquals("hello", JSONObject(request.body.readUtf8()).getJSONArray("messages").getJSONObject(0).getString("content"))
    }
    @Test fun explicitStopUsesRawRequestId() = server { s, base ->
        s.enqueue(response(answer).setBodyDelay(50, TimeUnit.MILLISECONDS))
        s.enqueue(MockResponse().setResponseCode(202))
        val pool = Executors.newFixedThreadPool(2)
        try {
            val job = Generation(base, pool)
            assertThrows(Exception::class.java) {
                job.run("local-model", listOf(Message("user", "hello")), 512) { job.stop() }
            }
            assertEquals("/v1/chat/completions", s.takeRequest(5, TimeUnit.SECONDS)!!.path)
            val cancel = s.takeRequest(5, TimeUnit.SECONDS)!!
            assertEquals("/requests/abc/cancel", cancel.path); assertEquals("POST", cancel.method)
            assertTrue(job.stopped.get())
        } finally { pool.shutdownNow() }
    }
    @Test fun stopBeforeStartingMakesNoRequest() = server { s, base ->
        val job = Generation(base, direct); job.stop()
        assertThrows(Exception::class.java) { job.run("local-model", listOf(Message("user", "hello")), 1) {} }
        assertEquals(0, s.requestCount)
    }
    @Test fun stopBeforeHeadersDoesNotDeliverText() = server { s, base ->
        s.enqueue(response(answer + finish + usage + "data: [DONE]\n\n").setHeadersDelay(300, TimeUnit.MILLISECONDS))
        val pool = Executors.newFixedThreadPool(3)
        try {
            val job = Generation(base, pool)
            val text = StringBuilder()
            val result = pool.submit<Completion> { job.run("local-model", listOf(Message("user", "hello")), 1) { text.append(it) } }
            assertNotNull(s.takeRequest(5, TimeUnit.SECONDS)); job.stop()
            assertThrows(ExecutionException::class.java) { result.get(5, TimeUnit.SECONDS) }
            assertEquals("", text.toString())
        } finally { pool.shutdownNow() }
    }
    @Test fun truncatedHttpIsNotSuccessful() = server { s, base ->
        s.enqueue(response(answer))
        assertThrows(Exception::class.java) { Generation(base, direct).run("local-model", listOf(Message("user", "hello")), 1) {} }
    }
    @Test fun admissionFailureShowsServerMessage() = server { s, base ->
        s.enqueue(MockResponse().setResponseCode(503).setBody("""{"error":{"message":"Queue is full"}}"""))
        val e = assertThrows(Exception::class.java) {
            Generation(base, direct).run("local-model", listOf(Message("user", "hello")), 1) {}
        }
        assertEquals("Queue is full", e.message)
    }
}
