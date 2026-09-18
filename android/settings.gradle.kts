pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven { url = uri("../GenUICraft/build/repo") }
        google()
        mavenCentral()
    }
}

rootProject.name = "GenUiCraftRenderer"
include(":app")
