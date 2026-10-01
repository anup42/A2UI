# Android Diagnostics Summary

artifact_dir: GenUICraft\validation\20260928_r64_qat_reexport\diagnostics\20260928_005441
serial: R3GL203AKSF
package: com.samsung.genuicraft
pids: (none detected)
device_info: GenUICraft\validation\20260928_r64_qat_reexport\diagnostics\20260928_005441\device.txt
raw_logcat: GenUICraft\validation\20260928_r64_qat_reexport\diagnostics\20260928_005441\logcat_raw.txt
filtered_logcat: GenUICraft\validation\20260928_r64_qat_reexport\diagnostics\20260928_005441\logcat_filtered.txt

```text
raw_log_lines: 108881
filtered_log_lines: 2026
levels: W=841, D=593, I=463, V=91, E=38
top_tags: PackageConfigPersister=826, SurfaceFlinger=192, FreecessHandler=158, WindowManager=108, AconfigPackage=100, SGM=46, adbd    =31, nativeloader=27, RenderEngine=24, ActivityManagerPerformance=23, ActivityManager=20, InputDispatcher=20

high_signal_excerpt_last_15_lines:
09-28 00:54:12.543  1987  2054 I HYPER-HAL: [RequestManager.cpp]releaseLocked(): Released ID : 58399031
09-28 00:54:12.544  3304  4227 D InetDiagMessage: Destroyed 0 sockets, proto=IPPROTO_TCP, family=AF_INET, states=14
09-28 00:54:12.544  4672  4672 D Settings: GET_req(/secure/one_handed_mode_timeout) userId:0, myUserId:0, isInSystemServer:false, isSelf:true, useCache:true, callingPackage:com.android.systemui
09-28 00:54:12.544  4672  4672 D Settings: GET_ret(/secure/one_handed_mode_timeout) value:null, userId:0, callingPackage:com.android.systemui (Cached)
09-28 00:54:12.544  3304  6869 D PkgPredictorService-SecIpmManager: recordRecentKillApp com.samsung.genuicraft
09-28 00:54:12.544  3304  6869 D PkgPredictorService-IpmNapPreloadController: recentKillApp: com.samsung.genuicraft
09-28 00:54:12.590  3304  6869 V ActivityManager: Got obituary of 24846:com.samsung.genuicraft
09-28 00:54:14.755  3304  5917 D FreecessHandler: skipping freeze com.samsung.genuicraft.test(10380) result : 2
09-28 00:54:20.765  3304  5917 D FreecessHandler: skipping freeze com.samsung.genuicraft.test(10380) result : 2
09-28 00:54:22.471  3304  4065 D ActivityManager: freezing 5613 com.samsung.genuicraft.test
09-28 00:54:26.780  3304  5917 D FreecessHandler: skipping freeze com.samsung.genuicraft.test(10380) result : 2
09-28 00:54:32.794  3304  5917 D FreecessHandler: skipping freeze com.samsung.genuicraft.test(10380) result : 2
09-28 00:54:38.808  3304  5917 D FreecessHandler: skipping freeze com.samsung.genuicraft.test(10380) result : 2
09-28 00:54:41.236 29235 29235 I adbd    : adbd service requested 'shell,v2,TERM=dumb,raw:pidof 'com.samsung.genuicraft''
09-28 00:54:41.675 29235 29235 I adbd    : adbd service requested 'shell,v2,TERM=dumb,raw:pidof 'com.samsung.genuicraft''
```
