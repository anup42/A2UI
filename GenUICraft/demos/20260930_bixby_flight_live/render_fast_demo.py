"""Speed up only the streaming portion of the annotated live Bixby video."""
from pathlib import Path
import argparse, subprocess, tempfile, json, hashlib
import imageio_ffmpeg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
source, output = args.source.resolve(), args.out.resolve()
assert source != output
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
ass = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1080
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Speed,Segoe UI,24,&H00448E19,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Clock,Segoe UI,13,&H00706255,&H00FFFFFF,&H00F7F5F4,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Main,Segoe UI,46,&H00332318,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Body,Segoe UI,27,&H00706255,&H00FFFFFF,&H00F7F5F4,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Vector,Arial,1,&H00448E19,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:00.00,0:00:48.80,Clock,,0,0,0,,{\\pos(951,103)}SOURCE TIME
Dialogue: 1,0:00:17.00,0:00:25.10,Speed,,0,0,0,,{\\pos(613,110)}GENERATION · 3×
Dialogue: 1,0:00:15.80,0:00:17.00,Speed,,0,0,0,,{\\pos(613,177)}SWITCH THE VIEW
Dialogue: 1,0:00:15.80,0:00:17.00,Main,,0,0,0,,{\\pos(613,225)\\fs38}Tap GenUICraft
Dialogue: 1,0:00:15.80,0:00:17.00,Vector,,0,0,0,,{\\pos(0,0)\\p1}m 492 254 l 513 243 513 251 596 251 596 257 513 257 513 265 492 254{\\p0}
Dialogue: 1,0:00:17.00,0:00:25.10,Speed,,0,0,0,,{\\pos(613,295)}LIVE A2UI EXPRESS
Dialogue: 1,0:00:17.00,0:00:25.10,Main,,0,0,0,,{\\pos(613,350)}From streaming\\Nto flight cards
Dialogue: 1,0:00:17.00,0:00:25.10,Body,,0,0,0,,{\\pos(613,500)}Streaming shown at 3×\\Nfor a shorter demo.
Dialogue: 1,0:00:17.00,0:00:25.10,Vector,,0,0,0,,{\\pos(0,0)\\p1}m 492 406 l 513 395 513 403 596 403 596 409 513 409 513 417 492 406{\\p0}
'''
graph = '''[0:v]split=3[a][b][c];
[a]trim=start=0:end=17,setpts=PTS-STARTPTS[v0];
[b]trim=start=17:end=41.3,setpts=(PTS-STARTPTS)/3,fps=30[v1];
[c]trim=start=41.3:end=65,setpts=PTS-STARTPTS[v2];
[v0][v1][v2]concat=n=3:v=1:a=0,drawbox=x=488:y=142:w=592:h=938:color=0xF4F5F7:t=fill:enable='gte(t,15.8)*lt(t,25.1)',subtitles=speed.ass,format=yuv420p[out]
'''
output.parent.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='bixby_fast_') as tmp:
    work=Path(tmp)
    (work/'speed.ass').write_text(ass,encoding='utf-8')
    (work/'speed.ffscript').write_text(graph,encoding='utf-8')
    subprocess.run([ffmpeg,'-y','-v','error','-i',str(source),'-filter_complex_script','speed.ffscript','-map','[out]','-an','-c:v','libx264','-preset','medium','-crf','17','-r','30','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],cwd=work,check=True)
subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(output),'-f','null','-'],check=True)
frames, seconds=imageio_ffmpeg.count_frames_and_secs(str(output))
assert frames==1464 and abs(seconds-48.8)<0.05,(frames,seconds)
metadata={'file':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'frames':frames,'duration_seconds':seconds,'source':source.name,'source_speed_sections':[{'from':0,'to':17,'speed':1},{'from':17,'to':41.3,'speed':3},{'from':41.3,'to':65,'speed':1}],'accelerated_output_seconds':[17,25.1],'timer':'elapsed time in the original recording','decode_errors':0}
output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata,indent=2))
