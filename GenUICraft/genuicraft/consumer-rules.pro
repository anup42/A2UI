# LiteRT-LM's JNI code looks up Kotlin classes and getters by their original names.
# A minified host otherwise removes SamplerConfig.getTopK() and aborts in
# nativeCreateConversation before GenUICraft can recover the conversion.
-keep class com.google.ai.edge.litertlm.** { *; }

# Benchmark getters are also accessed reflectively by the SDK.
-keepclassmembers class com.google.ai.edge.litertlm.Conversation {
    public com.google.ai.edge.litertlm.BenchmarkInfo getBenchmarkInfo();
}
-keepclassmembers class com.google.ai.edge.litertlm.BenchmarkInfo {
    public int getLastPrefillTokenCount();
    public int getLastDecodeTokenCount();
    public double getLastDecodeTokensPerSecond();
}
