package com.samsung.genuicraft

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.samsung.genuicraft.sdk.GenUiAction
import com.samsung.genuicraft.sdk.GenUiContent
import com.samsung.genuicraft.sdk.GenUiDocument
import com.samsung.genuicraft.sdk.GenUiRepairKind
import kotlinx.coroutines.delay

internal data class SdkGenerationTrace(
    val attempts: List<SdkGenerationAttempt> = emptyList(),
    val phase: SdkGenerationPhase = SdkGenerationPhase.IDLE,
    val finalIr: String? = null,
    val repairKind: GenUiRepairKind? = null,
    val error: String? = null,
    val warnings: List<String> = emptyList(),
)

internal data class SdkGenerationAttempt(
    val number: Int,
    /** Cumulative provider output. Each update replaces the previous snapshot. */
    val rawText: String = "",
    val complete: Boolean = false,
)

internal enum class SdkGenerationPhase {
    IDLE,
    GENERATING,
    REPAIRING,
    COMPLETE,
    FAILED,
    CANCELLED,
}

private enum class WorkspaceTab { IR, PREVIEW }

@Composable
internal fun SdkGenerationWorkspace(
    trace: SdkGenerationTrace,
    document: GenUiDocument?,
    modifier: Modifier = Modifier,
    metrics: @Composable () -> Unit = {},
    onAction: (GenUiAction) -> Unit,
) {
    var selectedTab by remember {
        mutableStateOf(
            if (trace.attempts.isNotEmpty() || document == null) WorkspaceTab.IR
            else WorkspaceTab.PREVIEW
        )
    }
    val hasAttempts = trace.attempts.isNotEmpty()
    LaunchedEffect(trace.phase, hasAttempts, document != null) {
        when {
            trace.phase == SdkGenerationPhase.COMPLETE && document != null ->
                selectedTab = WorkspaceTab.PREVIEW
            trace.phase == SdkGenerationPhase.GENERATING ||
                trace.phase == SdkGenerationPhase.REPAIRING -> selectedTab = WorkspaceTab.IR
            hasAttempts -> selectedTab = WorkspaceTab.IR
            document == null -> selectedTab = WorkspaceTab.IR
            else -> selectedTab = WorkspaceTab.PREVIEW
        }
    }
    val activeTab = if (selectedTab == WorkspaceTab.PREVIEW && document == null) {
        WorkspaceTab.IR
    } else {
        selectedTab
    }

    BoxWithConstraints(
        modifier = modifier
            .fillMaxSize()
            .testTag("sdk_generation_workspace")
            .semantics { contentDescription = "SDK generation workspace" },
    ) {
        val compact = maxWidth < 520.dp
        if (
            trace.phase == SdkGenerationPhase.IDLE &&
            trace.attempts.isEmpty() &&
            trace.finalIr == null &&
            document == null
        ) {
            EmptyWorkspace()
            return@BoxWithConstraints
        }
        Column(Modifier.fillMaxSize()) {
            PhaseTracker(trace, compact, document != null)
            TabRow(
                selectedTabIndex = if (activeTab == WorkspaceTab.IR) 0 else 1,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Tab(
                    selected = activeTab == WorkspaceTab.IR,
                    onClick = { selectedTab = WorkspaceTab.IR },
                    modifier = Modifier
                        .testTag("sdk_ir_tab")
                        .semantics { contentDescription = "IR output" },
                    text = { Text("Inspect IR") },
                )
                if (document != null) {
                    Tab(
                        selected = activeTab == WorkspaceTab.PREVIEW,
                        onClick = { selectedTab = WorkspaceTab.PREVIEW },
                        modifier = Modifier
                            .testTag("sdk_preview_tab")
                            .semantics { contentDescription = "Preview rendered UI" },
                        text = { Text("Preview") },
                    )
                }
            }

            when (activeTab) {
                WorkspaceTab.IR -> IrWorkspace(trace, compact, metrics)
                WorkspaceTab.PREVIEW -> document?.let { output ->
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .testTag("sdk_preview_content")
                            .verticalScroll(rememberScrollState())
                            .padding(top = 10.dp, bottom = 16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        GenUiContent(
                            document = output,
                            modifier = Modifier.fillMaxWidth(),
                            onAction = onAction,
                        )
                        metrics()
                    }
                }
            }
        }
    }
}

@Composable
private fun EmptyWorkspace() {
    Box(
        modifier = Modifier.fillMaxSize().padding(vertical = 12.dp),
        contentAlignment = Alignment.TopCenter,
    ) {
        OutlinedCard(Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(horizontal = 18.dp, vertical = 20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Your generated UI will appear here", style = MaterialTheme.typography.titleMedium)
                Text(
                    "Choose a sample or paste an answer, pick a model, then tap Generate UI.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    listOf("A2UI Express", "Recover", "Render").forEachIndexed { index, label ->
                        Surface(
                            modifier = Modifier.weight(1f),
                            shape = MaterialTheme.shapes.small,
                            color = MaterialTheme.colorScheme.surfaceVariant,
                        ) {
                            Text(
                                "${index + 1}  $label",
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PhaseTracker(trace: SdkGenerationTrace, compact: Boolean, hasDocument: Boolean) {
    val activeStep = when (trace.phase) {
        SdkGenerationPhase.GENERATING -> 0
        SdkGenerationPhase.REPAIRING -> 1
        SdkGenerationPhase.COMPLETE -> 3
        SdkGenerationPhase.IDLE -> if (hasDocument) 3 else -1
        else -> -1
    }
    val status = when (trace.phase) {
        SdkGenerationPhase.IDLE -> if (hasDocument) "Preview ready" else "Waiting to generate"
        SdkGenerationPhase.GENERATING -> "Generating IR · live output"
        SdkGenerationPhase.REPAIRING -> "Validating and repairing IR"
        SdkGenerationPhase.COMPLETE -> "IR ready"
        SdkGenerationPhase.FAILED -> "Generation failed · raw output preserved"
        SdkGenerationPhase.CANCELLED -> "Generation cancelled · partial output preserved"
    }
    val statusColor = when (trace.phase) {
        SdkGenerationPhase.FAILED -> MaterialTheme.colorScheme.error
        SdkGenerationPhase.CANCELLED -> MaterialTheme.colorScheme.tertiary
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    val labels = if (compact) listOf("Generate", "Recover", "Render")
    else listOf("Generate", "Validate & recover", "Render")

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 4.dp)
            .testTag("sdk_generation_phase")
            .semantics { contentDescription = "Generation phase: $status" },
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            labels.forEachIndexed { index, label ->
                val complete = activeStep > index
                val active = activeStep == index
                Text(
                    text = when {
                        complete -> "✓ $label"
                        active -> "• $label"
                        else -> label
                    },
                    modifier = Modifier.weight(1f).padding(vertical = 3.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = when {
                        complete -> MaterialTheme.colorScheme.primary
                        active -> MaterialTheme.colorScheme.onSecondaryContainer
                        else -> MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
            }
        }
        Text(status, style = MaterialTheme.typography.labelSmall, color = statusColor)
    }
}

@Composable
private fun IrWorkspace(
    trace: SdkGenerationTrace,
    compact: Boolean,
    metrics: @Composable () -> Unit,
) {
    val outerScroll = rememberScrollState()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(outerScroll)
            .padding(top = 10.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        if (trace.attempts.isEmpty() && trace.phase == SdkGenerationPhase.GENERATING) {
            CodePanel(
                title = "Generated IR · live",
                text = "",
                testTag = "sdk_live_ir",
                compact = compact,
                live = true,
            )
        }
        trace.attempts.forEachIndexed { index, attempt ->
            val live = trace.phase == SdkGenerationPhase.GENERATING &&
                index == trace.attempts.lastIndex && !attempt.complete
            val title = when {
                live -> "Generated IR · live"
                trace.attempts.size > 1 -> "Generated IR · attempt ${attempt.number}"
                else -> "Generated IR"
            }
            CodePanel(
                title = title,
                text = attempt.rawText,
                testTag = "sdk_generated_ir_${attempt.number}",
                liveTestTag = live,
                compact = compact,
                live = live,
                initiallyExpanded = live || trace.finalIr == null,
            )
        }

        trace.finalIr?.let { finalIr ->
            val title = when (trace.repairKind) {
                GenUiRepairKind.STRUCTURAL,
                GenUiRepairKind.GENERATED_DSL_REPAIR -> "Repaired IR"
                GenUiRepairKind.SOURCE_TEXT_FALLBACK -> "Source fallback"
                GenUiRepairKind.NONE, null -> "Validated IR"
            }
            CodePanel(
                title = title,
                text = finalIr,
                testTag = "sdk_repaired_ir",
                compact = compact,
                supportingText = finalIrSupportingText(trace.repairKind),
                initiallyExpanded = false,
            )
        }

        ConversionNotes(trace.warnings)

        if (
            trace.phase == SdkGenerationPhase.FAILED ||
            trace.phase == SdkGenerationPhase.CANCELLED
        ) {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.medium,
                color = if (trace.phase == SdkGenerationPhase.FAILED) {
                    MaterialTheme.colorScheme.errorContainer
                } else {
                    MaterialTheme.colorScheme.tertiaryContainer
                },
            ) {
                Text(
                    if (trace.phase == SdkGenerationPhase.FAILED) {
                        "The input and generated output are still available. Edit the input or try the run again."
                    } else {
                        "The run stopped safely. Partial generated output is still available to inspect or copy."
                    },
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        trace.error?.takeIf(String::isNotBlank)?.let { error ->
            CodePanel(
                title = if (trace.phase == SdkGenerationPhase.CANCELLED) {
                    "Cancellation details"
                } else {
                    "Error details"
                },
                text = error,
                testTag = "sdk_generation_error",
                compact = compact,
                error = true,
            )
        }

        if (
            trace.attempts.isEmpty() && trace.finalIr == null && trace.error == null &&
            trace.phase != SdkGenerationPhase.GENERATING
        ) {
            Text(
                "Generated IR will appear here as the model streams it.",
                modifier = Modifier.padding(horizontal = 4.dp, vertical = 12.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        metrics()
    }
}

@Composable
private fun CodePanel(
    title: String,
    text: String,
    testTag: String,
    compact: Boolean,
    live: Boolean = false,
    liveTestTag: Boolean = false,
    supportingText: String? = null,
    error: Boolean = false,
    initiallyExpanded: Boolean = true,
) {
    val clipboard = LocalClipboardManager.current
    val verticalScroll = rememberScrollState()
    val horizontalScroll = rememberScrollState()
    var expanded by remember(testTag, initiallyExpanded) { mutableStateOf(initiallyExpanded) }
    var copied by remember(text) { mutableStateOf(false) }
    val codeHeight = when {
        expanded && compact -> 320.dp
        expanded -> 420.dp
        compact -> 132.dp
        else -> 168.dp
    }

    LaunchedEffect(text.length, live) {
        if (live && text.isNotEmpty()) {
            withFrameNanos { }
            verticalScroll.scrollTo(verticalScroll.maxValue)
        }
    }
    LaunchedEffect(copied) {
        if (copied) {
            delay(1_200)
            copied = false
        }
    }

    OutlinedCard(
        modifier = Modifier
            .fillMaxWidth()
            .testTag(testTag)
            .semantics {
                contentDescription = "$title, ${text.length} characters"
            },
        colors = CardDefaults.outlinedCardColors(
            containerColor = if (error) {
                MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.24f)
            } else {
                MaterialTheme.colorScheme.surface
            }
        ),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(start = 12.dp, end = 4.dp, top = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleSmall)
                Text(
                    buildString {
                        append("${text.length} chars")
                        if (live) append(" · streaming")
                        supportingText?.let { append(" · ").append(it) }
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = if (error) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            TextButton(
                onClick = {
                    clipboard.setText(AnnotatedString(text))
                    copied = true
                },
                enabled = text.isNotEmpty(),
                modifier = Modifier.semantics { contentDescription = "Copy $title" },
            ) { Text(if (copied) "Copied" else "Copy") }
            TextButton(
                onClick = { expanded = !expanded },
                modifier = Modifier.semantics {
                    contentDescription = if (expanded) "Collapse $title" else "Expand $title"
                },
            ) { Text(if (expanded) "Hide code" else "Show code") }
        }
        if (expanded) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(codeHeight)
                    .then(if (liveTestTag) Modifier.testTag("sdk_live_ir") else Modifier),
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.34f),
            ) {
                SelectionContainer {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(verticalScroll)
                            .horizontalScroll(horizontalScroll)
                            .padding(12.dp),
                    ) {
                        Text(
                            text = text.ifEmpty { "Waiting for model output…" },
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace,
                                lineHeight = 18.sp,
                            ),
                            color = if (text.isEmpty()) MaterialTheme.colorScheme.onSurfaceVariant
                            else MaterialTheme.colorScheme.onSurface,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ConversionNotes(warnings: List<String>) {
    if (warnings.isEmpty()) return
    var expanded by remember { mutableStateOf(false) }
    OutlinedCard(Modifier.fillMaxWidth().testTag("sdk_conversion_notes")) {
        TextButton(
            onClick = { expanded = !expanded },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Conversion notes (${warnings.size})")
            Box(Modifier.weight(1f))
            Text(if (expanded) "Hide" else "Show")
        }
        if (expanded) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            SelectionContainer {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    warnings.forEach { note ->
                        Text("• $note", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}

private fun finalIrSupportingText(repairKind: GenUiRepairKind?): String = when (repairKind) {
    GenUiRepairKind.STRUCTURAL -> "bounded structural repair"
    GenUiRepairKind.GENERATED_DSL_REPAIR -> "generated-output DSL repair"
    GenUiRepairKind.SOURCE_TEXT_FALLBACK -> "generated output rejected"
    GenUiRepairKind.NONE, null -> "compiler accepted"
}
