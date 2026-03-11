import java.io.File
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.exists()) {
        file.inputStream().use { load(it) }
    }
}

fun parseDotEnv(file: File): Map<String, String> {
    val values = mutableMapOf<String, String>()
    file.forEachLine { rawLine ->
        val line = rawLine.trim()
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
        if (key.isEmpty()) {
            return@forEachLine
        }

        var value = normalized.substring(delimiter + 1).trim()
        if (
            (value.startsWith("\"") && value.endsWith("\"")) ||
                (value.startsWith("'") && value.endsWith("'"))
        ) {
            value = value.substring(1, value.length - 1)
        }
        values[key] = value
    }
    return values
}

fun escapeForBuildConfig(value: String): String {
    return value.replace("\\", "\\\\").replace("\"", "\\\"")
}

val dotEnvCandidates = listOf(
    rootProject.file(".env"),
    rootProject.file("../.env"),
    rootProject.file("../dataset/.env")
)

val geminiApiKey: String =
    (
        (project.findProperty("GEMINI_API_KEY") as? String)
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?: localProperties
                .getProperty("GEMINI_API_KEY")
                ?.trim()
                ?.takeIf { it.isNotBlank() }
            ?: dotEnvCandidates
                .asSequence()
                .filter { it.exists() }
                .mapNotNull { parseDotEnv(it)["GEMINI_API_KEY"]?.takeIf { key -> key.isNotBlank() } }
                .firstOrNull()
            ?: System.getenv("GEMINI_API_KEY")?.takeIf { it.isNotBlank() }
            ?: ""
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
        buildConfigField("String", "GEMINI_API_KEY", "\"${escapeForBuildConfig(geminiApiKey)}\"")
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
}
