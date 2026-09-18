# LiteRT-LM 0.15.0 benchmark getters are accessed reflectively because Kotlin metadata hides them.
-keepclassmembers class com.google.ai.edge.litertlm.Conversation {
    public com.google.ai.edge.litertlm.BenchmarkInfo getBenchmarkInfo();
}
-keepclassmembers class com.google.ai.edge.litertlm.BenchmarkInfo {
    public int getLastPrefillTokenCount();
    public int getLastDecodeTokenCount();
    public double getLastDecodeTokensPerSecond();
}
