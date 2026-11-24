package dev.llmruntime.chat

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.InputStreamReader
import java.io.StringReader
import java.nio.charset.CodingErrorAction

class ChatProtocolTest {
    private fun chunk(choices: String, usage: String = "null", model: String = "local-model") =
        """{"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,"model":"$model","choices":$choices,"usage":$usage}"""
    private val text = """[{"index":0,"delta":{"role":"assistant","content":"Hei 🌍"},"finish_reason":null}]"""
    private val finish = """[{"index":0,"delta":{},"finish_reason":"length"}]"""
    private val usage = """{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}"""
    private fun stream() = ChatStream("local-model", "abc") {}
    private fun rejected(action: () -> Unit) { assertThrows(Exception::class.java, action) }

    @Test fun successfulStreamRequiresFinishUsageDone() {
        val answer = StringBuilder()
        val s = ChatStream("local-model", "abc") { answer.append(it) }
        assertTrue(s.accept(chunk(text))); assertTrue(s.accept(chunk(finish)))
        assertTrue(s.accept(chunk("[]", usage))); assertFalse(s.accept("[DONE]"))
        assertEquals("Hei 🌍", answer.toString()); assertEquals(Completion("length", 3), s.completed())
    }
    @Test fun truncatedStreamIsFailure() { rejected { stream().completed() } }
    @Test fun doneBeforeUsageIsFailure() { val s = stream(); s.accept(chunk(finish)); rejected { s.accept("[DONE]") } }
    @Test fun wrongModelIsFailure() { rejected { stream().accept(chunk(text, model = "other")) } }
    @Test fun wrongRequestIdIsFailure() { rejected { stream().accept(chunk(text).replace("chatcmpl-abc", "chatcmpl-other")) } }
    @Test fun changingTimestampIsFailure() {
        val s = stream(); s.accept(chunk(text)); rejected { s.accept(chunk(finish).replace("\"created\":1", "\"created\":2")) }
    }
    @Test fun textAfterFinishIsFailure() { val s = stream(); s.accept(chunk(finish)); rejected { s.accept(chunk(text)) } }
    @Test fun invalidUsageIsFailure() {
        val s = stream(); s.accept(chunk(finish)); rejected { s.accept(chunk("[]", usage.replace(":5", ":6"))) }
    }
    @Test fun duplicateUsageIsFailure() {
        val s = stream(); s.accept(chunk(finish)); s.accept(chunk("[]", usage)); rejected { s.accept(chunk("[]", usage)) }
    }
    @Test fun errorIsFailure() { rejected { stream().accept("""{"error":{"message":"Cancelled","code":"request_cancelled"}}""") } }
    @Test fun unknownFinishIsFailure() { rejected { stream().accept(chunk(finish.replace("length", "cancelled"))) } }
    @Test fun numericTextIsFailure() { rejected { stream().accept(chunk(text.replace("\"Hei 🌍\"", "7"))) } }
    @Test fun sseSupportsCommentsMultilineCrLfAndIncompleteTail() {
        val events = mutableListOf<String>()
        readEvents(StringReader(":ping\r\ndata: first\r\ndata:second\r\n\r\ndata: third\r\rdata: unfinished")) {
            events += it; true
        }
        assertEquals(listOf("first\nsecond", "third"), events)
    }
    @Test fun utf8AcrossOneByteReads() {
        val bytes = "data: blåbær 🌍\n\n".toByteArray(Charsets.UTF_8)
        val input = object : ByteArrayInputStream(bytes) {
            override fun read(b: ByteArray, off: Int, len: Int) = super.read(b, off, minOf(len, 1))
        }
        val events = mutableListOf<String>()
        InputStreamReader(input, Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)).use {
            readEvents(it) { event -> events += event; true }
        }
        assertEquals(listOf("blåbær 🌍"), events)
    }
    @Test fun oversizedSseIsRejected() { rejected { readEvents(StringReader("data: " + "x".repeat(1_048_576))) { true } } }
    @Test fun onlyCommittedTurnsGoIntoFollowUp() {
        val c = Conversation()
        c.request("failed"); c.request("stopped")
        assertTrue(c.messages.isEmpty())
        c.complete("My name is Ada", "Hello Ada")
        assertEquals(listOf("user", "assistant", "user"), c.request("Who am I?").map { it.role })
        assertEquals("My name is Ada", c.messages.first().content)
    }
    @Test fun historyLimitsAndBlankPromptAreRejected() {
        val c = Conversation()
        rejected { c.request("  ") }; rejected { c.request("x".repeat(8193)) }
        repeat(32) { c.complete("hi", "hello") }
        rejected { c.request("next") }
    }
    @Test fun unicodeLimitCountsCodePoints() { assertEquals(1, Conversation().request("🌍".repeat(8192)).size) }
    @Test fun addressValidation() {
        assertEquals("http://10.0.2.2:8000", serverUrl(" http://10.0.2.2:8000/ "))
        listOf("file:///tmp/a", "http://user:secret@host", "http://host/path", "http://host?x=1", "http://host:0")
            .forEach { rejected { serverUrl(it) } }
    }
    @Test fun requestUsesExactApiContract() {
        val body = JSONObject(chatBody("local-model", listOf(Message("user", "blåbær")), 32))
        assertEquals(32, body.getInt("max_tokens")); assertEquals(0, body.getInt("temperature"))
        assertTrue(body.getBoolean("stream")); assertTrue(body.getJSONObject("stream_options").getBoolean("include_usage"))
        assertEquals("blåbær", body.getJSONArray("messages").getJSONObject(0).getString("content"))
        rejected { chatBody("local-model", emptyList(), 513) }
    }
}
