import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const skill='C:/Users/anupk/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const runtime='C:/Users/anupk/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE=process.execPath;
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
process.env.RUNTIME_PYTHON=runtime+'/python/python.exe';
const workspaceDir='C:/Users/anupk/Documents/git/A2UI/GenUICraft/presentations/20260930_bixby_demo';
const build=path.join(workspaceDir,'.build');
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const result=await finalizePresentation({
 workspaceDir,
 candidatePath:path.join(build,'candidate-notimer-embedded.pptx'),
 finalPath:path.join(workspaceDir,'output','Bixby_GenUICraft_Demo_NoTimer.pptx'),
 explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 fontPolicy:{basis:'design',families:['Arial']},
 verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation-notimer.json')
});
await fs.writeFile(path.join(build,'finalization-notimer-result.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
