plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

import java.util.Properties

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
val embeddedVertexExpressApiKey = resolveSecret(
    "VERTEX_EXPRESS_API_KEY",
    "GEMINI_VERTEX_EXPRESS_API_KEY"
).ifBlank {
    // User-requested fallback to keep Vertex Express key bundled in shared APK builds.
    "Ab8RN6IRHBflUl3jG3DkjYhmgUVMeKlxPjwVDaJhWcQun7RZ_A"
}
val embeddedVertexProjectId = resolveSecret(
    "VERTEX_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GCP_PROJECT_ID"
)
val embeddedVertexOauthAccessToken = resolveSecret(
    "VERTEX_OAUTH_ACCESS_TOKEN",
    "GOOGLE_OAUTH_ACCESS_TOKEN"
)

android {
    namespace = "com.samsung.genuicraft"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.samsung.genuicraft"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
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
    }

    buildTypes {
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

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
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

    debugImplementation(composeBom)
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:core-ktx:1.6.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.test.uiautomator:uiautomator:2.3.0")
}
