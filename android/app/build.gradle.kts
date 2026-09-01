plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    jacoco
}

import java.util.Properties
import groovy.json.JsonSlurper
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.gradle.testing.jacoco.tasks.JacocoReport

val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.exists()) {
        file.inputStream().use(::load)
    }
}

val dotEnvValues = linkedMapOf<String, String>().apply {
    val candidates = listOf(
        rootProject.file(".env"),
        rootProject.file("../.env"),
        rootProject.file("../dataset/.env")
    )
    candidates.firstOrNull { it.exists() }?.forEachLine { raw ->
        val line = raw.trim()
        if (line.isEmpty() || line.startsWith("#")) {
            return@forEachLine
        }
        val normalized = if (line.startsWith("export ")) {
            line.removePrefix("export ").trim()
        } else {
            line
        }
        val delimiter = normalized.indexOf('=')
        if (delimiter <= 0) {
            return@forEachLine
        }
        val key = normalized.substring(0, delimiter).trim()
        if (key.isBlank()) {
            return@forEachLine
        }
        var value = normalized.substring(delimiter + 1).trim()
        if (
            (value.startsWith("\"") && value.endsWith("\"")) ||
            (value.startsWith("'") && value.endsWith("'"))
        ) {
            value = value.substring(1, value.length - 1)
        }
        if (value.isNotBlank()) {
            this[key] = value
        }
    }
}

fun resolveSecret(vararg keys: String): String {
    for (key in keys) {
        val gradleProp = (findProperty(key) as? String)?.trim()
        if (!gradleProp.isNullOrBlank()) {
            return gradleProp
        }
        val localProp = localProperties.getProperty(key)?.trim()
        if (!localProp.isNullOrBlank()) {
            return localProp
        }
        val dotEnvValue = dotEnvValues[key]?.trim()
        if (!dotEnvValue.isNullOrBlank()) {
            return dotEnvValue
        }
        val envVar = System.getenv(key)?.trim()
        if (!envVar.isNullOrBlank()) {
            return envVar
        }
    }
    return ""
}

fun escapeForBuildConfig(value: String): String {
    return value
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
}

val embeddedStage2ApiKey = resolveSecret(
    "GEMINI_STAGE2_API_KEY",
    "GEMINI_RESPONSE_API_KEY",
    "GEMINI_API_KEY"
)
val embeddedStage3ApiKey = resolveSecret(
    "GEMINI_STAGE3_API_KEY",
    "GEMINI_IR_API_KEY",
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY"
)
val embeddedNewsApiKey = resolveSecret("NEWS_API_KEY")
val embeddedSerpApiKey = resolveSecret("SERPAPI_KEY")
val embeddedGoogleMapsApiKey = resolveSecret(
    "GOOGLE_MAPS_API_KEY",
    "GOOGLE_PLACES_API_KEY",
    "PLACES_API_KEY"
)
val embeddedVertexExpressApiKey = resolveSecret(
    "VERTEX_EXPRESS_API_KEY",
    "GEMINI_VERTEX_EXPRESS_API_KEY"
)
val embeddedVertexProjectId = resolveSecret(
    "VERTEX_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GCP_PROJECT_ID"
)
val embeddedVertexOauthAccessToken = resolveSecret(
    "VERTEX_OAUTH_ACCESS_TOKEN",
    "GOOGLE_OAUTH_ACCESS_TOKEN"
)
val embeddedAzureOpenAiApiKey = resolveSecret(
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_SUBSCRIPTION_KEY"
)
val embeddedAzureOpenAiEndpoint = resolveSecret(
    "AZURE_OPENAI_RESPONSES_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT"
).ifBlank {
    "https://genui1.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
}
val embeddedAzureOpenAiDeployment = resolveSecret(
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_MODEL"
).ifBlank {
    "gpt-5.4-mini"
}

val rendererCapabilitiesManifest =
    rootProject.layout.projectDirectory.file("../dataset/schema/renderer_capabilities.json")
val generatedRendererCapabilitiesDir =
    layout.buildDirectory.dir("generated/source/rendererCapabilities/main/kotlin")
val generatedIntentFixturesDir =
    layout.buildDirectory.dir("generated/assets/intentFixtures")

val generateRendererCapabilities by tasks.registering {
    group = "build setup"
    description = "Generates the Android renderer capability contract from the shared v2 manifest."
    inputs.file(rendererCapabilitiesManifest)
    outputs.dir(generatedRendererCapabilitiesDir)

    doLast {
        @Suppress("UNCHECKED_CAST")
        val manifest = JsonSlurper().parse(rendererCapabilitiesManifest.asFile) as Map<String, Any?>
        @Suppress("UNCHECKED_CAST")
        val types = manifest["types"] as List<Map<String, Any?>>
        @Suppress("UNCHECKED_CAST")
        val compatibilityAliases =
            manifest["compatibility_type_aliases"] as Map<String, Map<String, Any?>>
        @Suppress("UNCHECKED_CAST")
        val tableDomains = manifest["table_domains"] as Map<String, Any?>
        @Suppress("UNCHECKED_CAST")
        val actions = manifest["actions"] as List<Map<String, Any?>>
        @Suppress("UNCHECKED_CAST")
        val chartSubtypes = manifest["chart_subtypes"] as Map<String, Any?>

        fun quote(value: Any?): String = "\"" + value.toString()
            .replace("\\", "\\\\")
            .replace("\"", "\\\"") + "\""
        fun stringSet(values: Iterable<*>): String = values.joinToString(
            prefix = "setOf(",
            postfix = ")"
        ) { quote(it) }
        fun stringMap(values: Map<*, *>): String = values.entries.joinToString(
            prefix = "mapOf(",
            postfix = ")"
        ) { (key, value) -> "${quote(key)} to ${quote(value)}" }

        val typeAliases = linkedMapOf<String, String>()
        val consumedProps = linkedMapOf<String, List<*>>()
        types.forEach { type ->
            val runtimeKey = type["runtime_key"].toString()
            @Suppress("UNCHECKED_CAST")
            val aliases = type["aliases"] as List<*>
            aliases.forEach { alias -> typeAliases[alias.toString()] = runtimeKey }
            @Suppress("UNCHECKED_CAST")
            val props = type["consumed_props"] as List<*>
            consumedProps[runtimeKey] = props
        }
        val compatibilityDirections = compatibilityAliases.mapValues { (_, value) ->
            @Suppress("UNCHECKED_CAST")
            val props = value["props"] as Map<String, Any?>
            props["direction"].toString()
        }
        @Suppress("UNCHECKED_CAST")
        val domainAliases = tableDomains["aliases"] as Map<String, String>
        @Suppress("UNCHECKED_CAST")
        val chartAliases = chartSubtypes["aliases"] as Map<String, String>

        val generated = buildString {
            appendLine("package com.samsung.genuicraft.renderer.flat.capability")
            appendLine()
            appendLine("/** Generated from dataset/schema/renderer_capabilities.json. Do not edit. */")
            appendLine("internal object GeneratedRendererCapabilities {")
            appendLine("    const val VERSION: String = ${quote(manifest["version"])}")
            appendLine("    val canonicalTypes: Set<String> = ${stringSet(types.map { it["canonical"] })}")
            appendLine("    val runtimeTypes: Set<String> = ${stringSet(types.map { it["runtime_key"] })}")
            appendLine("    val typeAliases: Map<String, String> = ${stringMap(typeAliases)}")
            appendLine("    val compatibilityTypeDirections: Map<String, String> = ${stringMap(compatibilityDirections)}")
            appendLine("    val consumedProps: Map<String, Set<String>> = mapOf(")
            consumedProps.entries.forEachIndexed { index, (type, props) ->
                val suffix = if (index == consumedProps.size - 1) "" else ","
                appendLine("        ${quote(type)} to ${stringSet(props)}$suffix")
            }
            appendLine("    )")
            @Suppress("UNCHECKED_CAST")
            appendLine("    val tableDomains: Set<String> = ${stringSet(tableDomains["canonical"] as List<*>)}")
            @Suppress("UNCHECKED_CAST")
            appendLine("    val cardFirstTableDomains: Set<String> = ${stringSet(tableDomains["card_first"] as List<*>)}")
            appendLine("    val tableDomainAliases: Map<String, String> = ${stringMap(domainAliases)}")
            appendLine("    val actionNames: Set<String> = ${stringSet(actions.map { it["name"] })}")
            appendLine("    val actionRuntimeKeys: Set<String> = ${stringSet(actions.map { it["runtime_key"] })}")
            appendLine("    val actionRequired: Map<String, Set<String>> = mapOf(")
            actions.forEachIndexed { index, action ->
                @Suppress("UNCHECKED_CAST")
                val required = action["required"] as? List<*> ?: emptyList<Any?>()
                val suffix = if (index == actions.size - 1) "" else ","
                appendLine("        ${quote(action["runtime_key"])} to ${stringSet(required)}$suffix")
            }
            appendLine("    )")
            appendLine("    val actionRequiredAny: Map<String, List<Set<String>>> = mapOf(")
            actions.forEachIndexed { index, action ->
                @Suppress("UNCHECKED_CAST")
                val groups = action["required_any"] as? List<List<*>> ?: emptyList()
                val rendered = groups.joinToString(prefix = "listOf(", postfix = ")") { stringSet(it) }
                val suffix = if (index == actions.size - 1) "" else ","
                appendLine("        ${quote(action["runtime_key"])} to $rendered$suffix")
            }
            appendLine("    )")
            appendLine("    val safeUrlActions: Set<String> = ${stringSet(actions.filter { it["safe_url"] == true }.map { it["runtime_key"] })}")
            @Suppress("UNCHECKED_CAST")
            appendLine("    val chartSubtypes: Set<String> = ${stringSet(chartSubtypes["canonical"] as List<*>)}")
            appendLine("    val chartSubtypeAliases: Map<String, String> = ${stringMap(chartAliases)}")
            appendLine("}")
        }

        val output = generatedRendererCapabilitiesDir.get().asFile.resolve(
            "com/samsung/genuicraft/renderer/flat/capability/GeneratedRendererCapabilities.kt"
        )
        output.parentFile.mkdirs()
        output.writeText(generated, Charsets.UTF_8)
    }
}

val syncIntentFixtures by tasks.registering(Copy::class) {
    group = "build setup"
    description = "Copies the shared 32-intent canonical fixture corpus into Android assets."
    from(rootProject.file("../dataset/tests/fixtures/intent_flat_specs_v2.json"))
    into(generatedIntentFixturesDir)
}

android {
    namespace = "com.samsung.genuicraft"
    compileSdk = 35
    // Keep ordinary instrumentation on debug, but allow the model handoff
    // gate to target the credential-free, data-isolated judgeCapture package:
    //   gradlew -PandroidTestBuildType=judgeCapture ...
    testBuildType = providers.gradleProperty("androidTestBuildType")
        .orElse("debug")
        .get()

    defaultConfig {
        applicationId = "com.samsung.genuicraft"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "1.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField(
            "String",
            "GEMINI_STAGE2_API_KEY_DEFAULT",
            "\"${escapeForBuildConfig(embeddedStage2ApiKey)}\""
        )
        buildConfigField(
            "String",
            "GEMINI_STAGE3_API_KEY_DEFAULT",
            "\"${escapeForBuildConfig(embeddedStage3ApiKey)}\""
        )
        buildConfigField(
            "String",
            "NEWS_API_KEY_DEFAULT",
            "\"${escapeForBuildConfig(embeddedNewsApiKey)}\""
        )
        buildConfigField(
            "String",
            "SERPAPI_KEY_DEFAULT",
            "\"${escapeForBuildConfig(embeddedSerpApiKey)}\""
        )
        buildConfigField(
            "String",
            "GOOGLE_MAPS_API_KEY_DEFAULT",
            "\"${escapeForBuildConfig(embeddedGoogleMapsApiKey)}\""
        )
        buildConfigField(
            "String",
            "VERTEX_EXPRESS_API_KEY_DEFAULT",
            "\"${escapeForBuildConfig(embeddedVertexExpressApiKey)}\""
        )
        buildConfigField(
            "String",
            "VERTEX_OAUTH_ACCESS_TOKEN_DEFAULT",
            "\"${escapeForBuildConfig(embeddedVertexOauthAccessToken)}\""
        )
        buildConfigField(
            "String",
            "VERTEX_PROJECT_ID_DEFAULT",
            "\"${escapeForBuildConfig(embeddedVertexProjectId)}\""
        )
        buildConfigField(
            "String",
            "AZURE_OPENAI_API_KEY_DEFAULT",
            "\"${escapeForBuildConfig(embeddedAzureOpenAiApiKey)}\""
        )
        buildConfigField(
            "String",
            "AZURE_OPENAI_RESPONSES_ENDPOINT_DEFAULT",
            "\"${escapeForBuildConfig(embeddedAzureOpenAiEndpoint)}\""
        )
        buildConfigField(
            "String",
            "AZURE_OPENAI_DEPLOYMENT_DEFAULT",
            "\"${escapeForBuildConfig(embeddedAzureOpenAiDeployment)}\""
        )
    }

    buildTypes {
        create("judgeCapture") {
            initWith(getByName("debug"))
            applicationIdSuffix = ".judgecapture"
            versionNameSuffix = "-judge-capture"
            matchingFallbacks += listOf("debug")
            isDebuggable = true
            // The capture-only package must never carry service credentials.
            buildConfigField("String", "GEMINI_STAGE2_API_KEY_DEFAULT", "\"\"")
            buildConfigField("String", "GEMINI_STAGE3_API_KEY_DEFAULT", "\"\"")
            buildConfigField("String", "NEWS_API_KEY_DEFAULT", "\"\"")
            buildConfigField("String", "SERPAPI_KEY_DEFAULT", "\"\"")
            buildConfigField("String", "GOOGLE_MAPS_API_KEY_DEFAULT", "\"\"")
            buildConfigField("String", "VERTEX_EXPRESS_API_KEY_DEFAULT", "\"\"")
            buildConfigField("String", "VERTEX_OAUTH_ACCESS_TOKEN_DEFAULT", "\"\"")
            buildConfigField("String", "VERTEX_PROJECT_ID_DEFAULT", "\"\"")
            buildConfigField("String", "AZURE_OPENAI_API_KEY_DEFAULT", "\"\"")
            buildConfigField(
                "String",
                "AZURE_OPENAI_RESPONSES_ENDPOINT_DEFAULT",
                "\"\""
            )
            buildConfigField("String", "AZURE_OPENAI_DEPLOYMENT_DEFAULT", "\"\"")
        }
        debug {
            enableUnitTestCoverage = true
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // LiteRT-LM GPU deployment uses the matching official GPU/OpenCL
    // accelerators.  Do not package the older CL-GL accelerator, which would
    // otherwise win LiteRT's runtime registry lookup and mix ABIs.
    packaging {
        jniLibs {
            excludes += "**/libLiteRtClGlAccelerator.so"
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    testOptions {
        unitTests.all {
            // Forward the routing-golden record flag into the test JVM so
            // `-DflatRoutingGolden.record=true` works from the command line.
            System.getProperty("flatRoutingGolden.record")?.let { value ->
                it.systemProperty("flatRoutingGolden.record", value)
            }
        }
    }

    sourceSets.named("main") {
        java.srcDir(generatedRendererCapabilitiesDir)
        assets.srcDir(generatedIntentFixturesDir)
    }
}

jacoco {
    toolVersion = "0.8.12"
}

val rendererCoverageClasses = files(
    fileTree(layout.buildDirectory.dir("tmp/kotlin-classes/debug")) {
        include("com/samsung/genuicraft/pipeline/FlatSpecContract*")
        include("com/samsung/genuicraft/pipeline/FlatSpecIngestor*")
        include("com/samsung/genuicraft/renderer/FlatRendererApi*")
        include("com/samsung/genuicraft/renderer/FlatSpecRenderer*")
        include("com/samsung/genuicraft/renderer/flat/**")
        exclude("**/*ComposableSingletons*")
    },
    fileTree(layout.buildDirectory.dir("intermediates/javac/debug/classes")) {
        include("com/samsung/genuicraft/pipeline/FlatSpecContract*")
        include("com/samsung/genuicraft/pipeline/FlatSpecIngestor*")
        include("com/samsung/genuicraft/renderer/FlatRendererApi*")
        include("com/samsung/genuicraft/renderer/FlatSpecRenderer*")
        include("com/samsung/genuicraft/renderer/flat/**")
    }
)

val rendererCoverageExecutionData = fileTree(layout.buildDirectory) {
    include("outputs/unit_test_code_coverage/debugUnitTest/testDebugUnitTest.exec")
    include("jacoco/testDebugUnitTest.exec")
}

tasks.register<JacocoReport>("rendererCoverageReport") {
    group = "verification"
    description = "Runs renderer unit tests and writes the renderer-focused JaCoCo HTML/XML report."
    dependsOn("testDebugUnitTest")
    classDirectories.setFrom(rendererCoverageClasses)
    sourceDirectories.setFrom(
        files("src/main/java", generatedRendererCapabilitiesDir)
    )
    executionData.setFrom(rendererCoverageExecutionData)
    reports {
        html.required.set(true)
        xml.required.set(true)
        csv.required.set(false)
    }
}

tasks.named("preBuild").configure {
    dependsOn(generateRendererCapabilities)
    dependsOn(syncIntentFixtures)
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

val requiredLauncherIconFiles = listOf(
    "src/main/res/mipmap-anydpi-v26/ic_launcher.xml",
    "src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml",
    "src/main/res/drawable/ic_launcher_background.xml",
    "src/main/res/drawable/ic_launcher_foreground.xml",
    "src/main/res/mipmap-mdpi/ic_launcher.png",
    "src/main/res/mipmap-mdpi/ic_launcher_round.png",
    "src/main/res/mipmap-hdpi/ic_launcher.png",
    "src/main/res/mipmap-hdpi/ic_launcher_round.png",
    "src/main/res/mipmap-xhdpi/ic_launcher.png",
    "src/main/res/mipmap-xhdpi/ic_launcher_round.png",
    "src/main/res/mipmap-xxhdpi/ic_launcher.png",
    "src/main/res/mipmap-xxhdpi/ic_launcher_round.png",
    "src/main/res/mipmap-xxxhdpi/ic_launcher.png",
    "src/main/res/mipmap-xxxhdpi/ic_launcher_round.png"
)

tasks.register("verifyLauncherIconAssets") {
    group = "verification"
    description = "Fails build if required launcher icon files are missing."
    doLast {
        val missingFiles = requiredLauncherIconFiles.filterNot { relativePath ->
            project.file(relativePath).isFile
        }
        if (missingFiles.isNotEmpty()) {
            throw org.gradle.api.GradleException(
                "Missing launcher icon asset(s):\n" +
                    missingFiles.joinToString(separator = "\n") { " - $it" }
            )
        }
    }
}

tasks.register("verifyGenUiPromptMirror") {
    group = "verification"
    description = "Fails build if app GenUI prompt and dataset mirror prompt drift."
    doLast {
        val appPrompt = project.file("src/main/assets/pipeline_prompts/genui_gen.md")
        val datasetPrompt = rootProject.file("../dataset/prompts/genui_gen.md")
        if (!appPrompt.isFile) {
            throw org.gradle.api.GradleException("Missing app prompt file: ${appPrompt.path}")
        }
        if (!datasetPrompt.isFile) {
            throw org.gradle.api.GradleException("Missing dataset mirror prompt file: ${datasetPrompt.path}")
        }
        val appText = appPrompt.readText(Charsets.UTF_8).replace("\r\n", "\n").trim()
        val datasetText = datasetPrompt.readText(Charsets.UTF_8).replace("\r\n", "\n").trim()
        if (appText != datasetText) {
            throw org.gradle.api.GradleException(
                "Prompt drift detected between app and dataset mirror.\n" +
                    "Canonical: ${appPrompt.path}\n" +
                    "Mirror: ${datasetPrompt.path}\n" +
                    "Sync flow: update app prompt first, then mirror in same change."
            )
        }
    }
}

tasks.named("preBuild").configure {
    dependsOn("verifyLauncherIconAssets")
    dependsOn("verifyGenUiPromptMirror")
}

// UTP connected instrumentation runs can leave the target debug package uninstalled
// after tests complete. Reinstall debug so launcher icon/app stay visible.
tasks.matching { task ->
    task.name.startsWith("connected") && task.name.endsWith("AndroidTest")
}.configureEach {
    finalizedBy("installDebug")
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.10.01")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("androidx.webkit:webkit:1.12.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("io.coil-kt:coil-compose:2.7.0")
    implementation("io.coil-kt:coil-svg:2.7.0")
    val patchedLiteRtLmAar = rootProject.projectDir.parentFile.resolve(
        "working_dir/litertlm-android-0.16.1-gpu-fixed-with-provider-v6.aar"
    )
    if (patchedLiteRtLmAar.isFile) {
        implementation(files(patchedLiteRtLmAar))
    } else {
        // Keep clean checkouts buildable; the patched AAR enables the optional
        // v0.16.1 GPU-provider path documented under android/tools.
        implementation("com.google.ai.edge.litertlm:litertlm-android:0.15.0")
    }

    debugImplementation(composeBom)
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:core-ktx:1.6.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.test.uiautomator:uiautomator:2.3.0")
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
}
