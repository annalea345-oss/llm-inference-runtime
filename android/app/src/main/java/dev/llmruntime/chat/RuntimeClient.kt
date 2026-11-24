package dev.llmruntime.chat

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.io.InputStreamReader
import java.io.Reader
import java.nio.charset.CodingErrorAction
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean

class Generation(private val base: String, private val cancellation: Executor) {
    val stopped = AtomicBoolean(false)
    private val cancelSent = AtomicBoolean(false)
    @Volatile private var connection: HttpURLConnection? = null
    @Volatile private var requestId: String? = null
    fun stop() {
        stopped.set(true)
        cancelKnownRequest()
        // Never block the UI waiting for a socket or a model step.
        cancellation.execute { connection?.disconnect() }
    }
    private fun cancelKnownRequest() {
        val id = requestId ?: return
        if (!cancelSent.compareAndSet(false, true)) return
        cancellation.execute {
            var conn: HttpURLConnection? = null
            try {
                conn = connect("$base/requests/$id/cancel", "POST")
                conn.readTimeout = 5000
                conn.responseCode // 404 also means the request is no longer live.
            } catch (_: Exception) { /* Closing the stream is the fallback. */ }
            finally { conn?.disconnect() }
        }
    }
    fun run(model: String, messages: List<Message>, limit: Int, text: (String) -> Unit): Completion {
        val conn = connect("$base/v1/chat/completions", "POST")
        connection = conn
        try {
            check(!stopped.get()) { "Stopped" }
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            conn.setRequestProperty("Accept", "text/event-stream")
            val bytes = chatBody(model, messages, limit).toByteArray(Charsets.UTF_8)
            conn.setFixedLengthStreamingMode(bytes.size)
            conn.outputStream.use { it.write(bytes) }
            checkResponse(conn)
            val id = conn.getHeaderField("X-Request-ID") ?: error("Missing request ID.")
            require(Regex("[A-Za-z0-9_-]{1,128}").matches(id)) { "Invalid request ID." }
            requestId = id
            if (stopped.get()) { cancelKnownRequest(); error("Stopped") }
            require(conn.contentType?.substringBefore(';')?.trim() == "text/event-stream") { "Expected an event stream." }
            val stream = ChatStream(model, id) { if (!stopped.get()) text(it) }
            val decoder = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
            InputStreamReader(conn.inputStream, decoder).use { readEvents(it, stream::accept) }
            check(!stopped.get()) { "Stopped" }
            return stream.completed()
        } finally { conn.disconnect(); connection = null }
    }
}

internal fun connect(url: String, method: String): HttpURLConnection =
    (URL(url).openConnection() as HttpURLConnection).apply {
        requestMethod = method
        connectTimeout = 5000
        readTimeout = 120_000
        instanceFollowRedirects = false
        useCaches = false
    }

internal fun checkResponse(conn: HttpURLConnection) {
    val code = conn.responseCode
    if (code !in 200..299) {
        val raw = conn.errorStream?.bufferedReader()?.use { readLimited(it, 65536) }
        val message = try { JSONObject(raw ?: "{}").getJSONObject("error").getString("message") }
            catch (_: Exception) { "Server returned HTTP $code." }
        error(message)
    }
}

fun discover(base: String): String {
    val conn = connect("$base/v1/models", "GET")
    conn.readTimeout = 5000
    try {
        checkResponse(conn)
        val raw = conn.inputStream.bufferedReader().use { readLimited(it, 65536) }
        val models = JSONObject(raw).getJSONArray("data")
        require(models.length() == 1) { "Expected this runtime's single model." }
        return models.getJSONObject(0).getString("id").also { require(it.isNotBlank()) }
    } finally { conn.disconnect() }
}

private fun readLimited(reader: Reader, limit: Int): String {
    val result = StringBuilder()
    val buffer = CharArray(4096)
    while (true) {
        val count = reader.read(buffer)
        if (count < 0) return result.toString()
        require(result.length + count <= limit) { "Server response is too large." }
        result.append(buffer, 0, count)
    }
}
