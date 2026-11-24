package dev.llmruntime.chat

import android.app.AlertDialog
import android.content.res.ColorStateList
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.lifecycle.ViewModelProvider

/** Native presentation only; all generation and accepted history remain in ChatModel. */
class MainActivity : ComponentActivity() {
    private lateinit var state: ChatModel
    private lateinit var prompt: EditText
    private lateinit var status: TextView
    private lateinit var messages: LinearLayout
    private lateinit var welcome: LinearLayout
    private lateinit var welcomeConnection: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var send: Button
    private lateinit var connect: Button
    private lateinit var settings: Button
    private lateinit var clear: Button
    private val replies = mutableListOf<Pair<TextView, TextView>>()
    private var firstRender = true
    private var settingsDialog: AlertDialog? = null
    private var detailsDialog: AlertDialog? = null
    private fun dp(n: Int) = (n * resources.displayMetrics.density).toInt()
    private fun color(id: Int) = getColor(id)
    private fun column() = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    private fun rounded(fill: Int, radius: Int = 18, border: Boolean = false) = GradientDrawable().apply {
        setColor(color(fill)); cornerRadius = dp(radius).toFloat()
        if (border) setStroke(dp(1), color(R.color.line))
    }
    private fun text(value: String, size: Float = 16f, muted: Boolean = false) = TextView(this).apply {
        text = value; textSize = size; setTextColor(color(if (muted) R.color.muted else R.color.ink))
        setLineSpacing(dp(3).toFloat(), 1f)
    }
    private fun label(id: Int, size: Float = 16f, muted: Boolean = false) = text(getString(id), size, muted)
    private fun button(id: Int, primary: Boolean = false) = Button(this).apply {
        setText(id); isAllCaps = false; textSize = 14f; minHeight = dp(48); minimumHeight = dp(48)
        setPadding(dp(16), dp(8), dp(16), dp(8))
        setTextColor(color(if (primary) R.color.on_accent else R.color.accent))
        background = RippleDrawable(ColorStateList.valueOf(color(R.color.line)),
            rounded(if (primary) R.color.accent else R.color.tint, 16), null)
    }
    private fun space(parent: LinearLayout, height: Int) { parent.addView(View(this), LinearLayout.LayoutParams(1, dp(height))) }
    private fun enabled(view: View, value: Boolean) { view.isEnabled = value; view.alpha = if (value) 1f else 0.45f }
    private fun onEdit(edit: EditText, change: () -> Unit) {
        edit.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) { change() }
        })
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        state = ViewModelProvider(this)[ChatModel::class.java]
        val root = column().apply { setBackgroundColor(color(R.color.page)) }
        root.setOnApplyWindowInsetsListener { view, insets ->
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars() or android.view.WindowInsets.Type.ime())
                view.setPadding(dp(16) + bars.left, dp(8) + bars.top, dp(16) + bars.right, dp(8) + bars.bottom)
            } else {
                @Suppress("DEPRECATION")
                view.setPadding(dp(16) + insets.systemWindowInsetLeft, dp(8) + insets.systemWindowInsetTop,
                    dp(16) + insets.systemWindowInsetRight, dp(8) + insets.systemWindowInsetBottom)
            }
            insets
        }
        val header = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        header.addView(label(R.string.app_name, 22f).apply { setTypeface(typeface, Typeface.BOLD) }, LinearLayout.LayoutParams(0, -2, 1f))
        clear = button(R.string.new_chat).apply { setOnClickListener { state.clear(); prompt.setText("") } }
        settings = button(R.string.settings).apply { setOnClickListener { showSettings() } }
        header.addView(clear)
        header.addView(settings, LinearLayout.LayoutParams(-2, -2).apply { marginStart = dp(8) })
        root.addView(header)
        status = label(R.string.not_connected, 13f, true).apply {
            minHeight = dp(48); gravity = Gravity.CENTER_VERTICAL
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
            setOnClickListener { showDetails() }
        }
        root.addView(status)
        val content = column()
        welcome = column().apply { setPadding(dp(12), dp(24), dp(12), dp(24)) }
        welcome.addView(label(R.string.welcome_title, 30f).apply { setTypeface(typeface, Typeface.BOLD) })
        space(welcome, 12)
        welcome.addView(label(R.string.welcome_body, 16f, true))
        space(welcome, 24)
        welcomeConnection = column().apply {
            background = rounded(R.color.surface, border = true); setPadding(dp(16), dp(16), dp(16), dp(16))
        }
        welcomeConnection.addView(label(R.string.connection_help, 14f, true))
        space(welcomeConnection, 14)
        connect = button(R.string.connect, true).apply { setOnClickListener { state.connect(state.base) } }
        welcomeConnection.addView(connect, LinearLayout.LayoutParams(-1, -2))
        welcome.addView(welcomeConnection)
        space(welcome, 16)
        listOf(R.string.example_one to R.string.example_one_prompt, R.string.example_two to R.string.example_two_prompt).forEach { (title, value) ->
            welcome.addView(button(title).apply { setOnClickListener {
                prompt.setText(value); prompt.setSelection(prompt.length()); prompt.requestFocus()
            } }, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(10) })
        }
        content.addView(welcome)
        messages = column()
        content.addView(messages)
        scroll = ScrollView(this).apply {
            addView(content); isFillViewport = true; clipToPadding = false; setPadding(0, 0, 0, dp(12))
        }
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        val composer = LinearLayout(this).apply {
            gravity = Gravity.BOTTOM; background = rounded(R.color.surface, 24, true)
            setPadding(dp(10), dp(8), dp(8), dp(8))
        }
        prompt = EditText(this).apply {
            setHint(R.string.message_hint); contentDescription = getString(R.string.message)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            maxLines = 3; minHeight = dp(48); textSize = 16f; setTextColor(color(R.color.ink))
            setHintTextColor(color(R.color.muted)); background = null
            setPadding(dp(6), dp(10), dp(6), dp(10)); setText(state.draft)
        }
        composer.addView(prompt, LinearLayout.LayoutParams(0, -2, 1f))
        send = button(R.string.send, true).apply { setOnClickListener {
            if (state.busy) state.stop() else {
                state.send(prompt.text.toString(), state.limit)
                if (state.busy) {
                    prompt.setText(state.draft)
                    (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).hideSoftInputFromWindow(prompt.windowToken, 0)
                }
            }
        } }
        composer.addView(send)
        root.addView(composer)
        root.addView(label(R.string.session_note, 11f, true).apply { gravity = Gravity.CENTER; setPadding(0, dp(8), 0, 0) })
        setContentView(root)
        onEdit(prompt) { state.draft = prompt.text.toString(); renderControls() }
        state.changed = ::render
        render()
    }

    private fun addTurn(turn: ChatTurn) {
        fun bubble(value: String, user: Boolean): LinearLayout {
            val card = column().apply {
                background = rounded(if (user) R.color.accent else R.color.surface, 20, !user)
                setPadding(dp(16), dp(12), dp(16), dp(12))
            }
            val name = label(if (user) R.string.you else R.string.assistant, 11f, !user).apply {
                if (user) setTextColor(color(R.color.on_accent))
                setTypeface(typeface, Typeface.BOLD)
            }
            card.addView(name)
            space(card, 4)
            val body = text(value).apply {
                id = if (user) R.id.user_message else R.id.assistant_reply
                setTextIsSelectable(true)
                if (user) setTextColor(color(R.color.on_accent))
            }
            card.addView(body)
            val row = LinearLayout(this).apply { gravity = if (user) Gravity.END else Gravity.START }
            row.addView(card, LinearLayout.LayoutParams(-1, -2).apply {
                if (user) marginStart = dp(48) else marginEnd = dp(20)
            })
            messages.addView(row, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(12) })
            return card
        }
        bubble(turn.user, true)
        val assistant = bubble(turn.reply, false)
        val body = assistant.findViewById<TextView>(R.id.assistant_reply)
        val note = text("", 12f, true).apply { setPadding(0, dp(8), 0, 0) }
        assistant.addView(note)
        replies += body to note
    }

    private fun render() {
        status.setText(when {
            state.status == "Connecting…" -> R.string.connecting
            state.status == "Generating…" -> R.string.writing
            state.status == "Stopping…" -> R.string.stopping
            state.status.startsWith("Could not connect:") -> R.string.connection_failed
            state.status.startsWith("Request failed:") -> R.string.request_failed
            state.status.startsWith("Stopped") -> R.string.stopped
            state.status.endsWith("output limit reached") -> R.string.length_status
            state.status.contains("output tokens") || state.status.startsWith("Connected ·") -> R.string.connected
            state.model == null -> R.string.not_connected
            else -> R.string.request_failed
        })
        val atBottom = scroll.getChildAt(0).height - scroll.height - scroll.scrollY < dp(80)
        val newTurn = replies.size < state.turns.size
        val cleared = replies.size > state.turns.size
        if (cleared) { messages.removeAllViews(); replies.clear() }
        while (replies.size < state.turns.size) addTurn(state.turns[replies.size])
        var changed = newTurn
        state.turns.forEachIndexed { i, turn ->
            val (body, note) = replies[i]
            val value = turn.reply.ifEmpty { if (turn.state == ReplyState.WRITING) getString(R.string.writing) else getString(R.string.empty_note) }
            if (body.text.toString() != value) { body.text = value; changed = true }
            val noteId = when (turn.state) {
                ReplyState.LIMIT -> R.string.limit_note
                ReplyState.STOPPED -> R.string.stopped_note
                ReplyState.FAILED -> R.string.failed_note
                else -> null
            }
            val noteValue = noteId?.let { getString(it) } ?: ""
            if (note.text.toString() != noteValue) { note.text = noteValue; changed = true }
            note.visibility = if (noteId == null) View.GONE else View.VISIBLE
        }
        welcome.visibility = if (state.turns.isEmpty()) View.VISIBLE else View.GONE
        welcomeConnection.visibility = if (state.model == null) View.VISIBLE else View.GONE
        renderControls()
        if (cleared) scroll.post { scroll.scrollTo(0, 0) }
        if (firstRender) {
            firstRender = false
            scroll.post {
                if (state.followLatest) scroll.fullScroll(View.FOCUS_DOWN) else scroll.scrollTo(0, state.scrollPosition)
            }
        } else if (changed && (atBottom || newTurn)) scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun renderControls() {
        enabled(settings, !state.busy); enabled(clear, !state.busy && state.turns.isNotEmpty())
        enabled(connect, !state.busy); prompt.isEnabled = !state.busy
        send.setText(if (state.busy && state.model != null) R.string.stop else R.string.send)
        enabled(send, if (state.busy) state.status == "Generating…" else state.model != null && prompt.text.isNotBlank())
    }

    private fun showDetails() {
        detailsDialog = AlertDialog.Builder(this).setTitle(R.string.status_details)
            .setMessage(getString(R.string.details_body, state.base, state.model ?: getString(R.string.no_model), state.status))
            .setPositiveButton(R.string.close, null).apply {
                if (!state.busy && state.model == null) setNeutralButton(R.string.connect) { _, _ -> state.connect(state.base) }
            }.show()
    }

    private fun showSettings() {
        if (state.busy) return
        val form = column().apply { setPadding(dp(24), dp(8), dp(24), dp(8)) }
        form.addView(label(R.string.settings_intro, 14f, true)); space(form, 20)
        form.addView(label(R.string.server_label, 16f))
        val address = EditText(this).apply {
            contentDescription = getString(R.string.server_address); setSingleLine()
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setText(state.base)
        }
        onEdit(address) { address.error = null }
        form.addView(address); form.addView(label(R.string.server_help, 12f, true)); space(form, 20)
        form.addView(label(R.string.reply_length, 16f)); form.addView(label(R.string.reply_length_help, 12f, true))
        val presets = RadioGroup(this)
        val choices = listOf(R.string.short_reply to "64", R.string.medium_reply to "192", R.string.long_reply to "512")
        val radios = choices.map { (label, value) -> RadioButton(this).apply {
            id = View.generateViewId(); setText(label); tag = value; minHeight = dp(48)
            presets.addView(this)
        } }
        radios.firstOrNull { it.tag == state.limit }?.let { presets.check(it.id) }
        form.addView(presets)
        val advanced = column()
        advanced.addView(label(R.string.token_help, 12f, true))
        val exact = EditText(this).apply {
            contentDescription = getString(R.string.output_token_limit); setSingleLine()
            inputType = InputType.TYPE_CLASS_NUMBER; setText(state.limit)
        }
        advanced.addView(exact); advanced.visibility = View.GONE
        form.addView(button(R.string.advanced).apply { setOnClickListener {
            advanced.visibility = if (advanced.visibility == View.GONE) View.VISIBLE else View.GONE
        } })
        form.addView(advanced)
        presets.setOnCheckedChangeListener { _, id -> radios.firstOrNull { it.id == id }?.let { exact.setText(it.tag as String) } }
        onEdit(exact) {
            exact.error = null
            val checked = radios.firstOrNull { it.id == presets.checkedRadioButtonId }
            if (checked != null && checked.tag != exact.text.toString()) presets.clearCheck()
        }
        val dialog = AlertDialog.Builder(this).setTitle(R.string.settings)
            .setView(ScrollView(this).apply { addView(form) })
            .setNegativeButton(R.string.cancel, null)
            .setPositiveButton(if (state.model == null) R.string.save_connect else R.string.save, null).create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val target = try { serverUrl(address.text.toString()) } catch (e: Exception) { address.error = e.message; return@setOnClickListener }
                val tokens = exact.text.toString().toIntOrNull()
                if (tokens == null || tokens !in 1..512) {
                    advanced.visibility = View.VISIBLE; exact.error = getString(R.string.invalid_limit); return@setOnClickListener
                }
                state.limit = tokens.toString()
                dialog.dismiss()
                if (target != state.base || state.model == null) state.connect(target)
            }
        }
        settingsDialog = dialog; dialog.setOnDismissListener { settingsDialog = null }; dialog.show()
    }

    override fun onDestroy() {
        settingsDialog?.dismiss()
        detailsDialog?.dismiss()
        state.draft = prompt.text.toString(); state.scrollPosition = scroll.scrollY
        state.followLatest = scroll.getChildAt(0).height - scroll.height - scroll.scrollY < dp(80)
        state.changed = null
        super.onDestroy()
    }
}
