param([ValidateSet('test','execute','cleanup')][string]$Mode='test')
$ErrorActionPreference='Stop'
$dockerExe='__DOCKER_EXE__'
$runDir=$PSScriptRoot
$projectRoot='__PROJECT_ROOT__'
$sourceDir=Join-Path $projectRoot 'frontier_research'
$envFile=Join-Path $projectRoot 'outputs/frontier_agent_team_setup_20261027_01/secrets/runtime.env'
$name='frontier-local-'+$Mode+'-'+[guid]::NewGuid().ToString('N').Substring(0,12)
if($Mode -eq 'execute' -and (Test-Path -LiteralPath (Join-Path $runDir 'STARTED.json'))) {throw 'This attempt already started; preserve it and prepare a new directory.'}
$dockerArgs=@('run','--rm','--init','--name',$name,
 '--mount',"type=bind,source=$sourceDir,target=/source,readonly",
 '--mount',"type=bind,source=$runDir,target=/trial",
 '--tmpfs','/workspace:rw,nosuid,size=512m,mode=1777',
 '-e','APODEX_RUN_DIR=/trial/native','-e','APODEX_SPILL_DIR=/trial/native/spill',
 '-e','FRONTIER_AGENT_OUTPUTS_DIR=/trial/native/outputs','-e','PYTHONPATH=/trial','-w','/trial')
if($Mode -eq 'test') {
 $dockerArgs+=@('--network','none','-e','DEEPSEEK_API_KEY=offline-unused',
  '-e','OPENAI_API_KEY=offline-unused','-e','OPENAI_MODEL=deepseek-flash',
  '-e','OPENAI_BASE_URL=https://api.deepseek.com','-e','OPENAI_PROVIDER=openai',
  '-e','SERPER_API_KEY=offline-unused','-e','JINA_API_KEY=offline-unused',
  '-e','E2B_API_KEY=offline-unused','-e','SANDBOX_BACKEND=e2b')
 $command='cp -a /source/. /app/ && exec python -B -m unittest test_linux_adapter.LinuxChecks test_budget_adapter.BudgetChecks test_verified_pricing.VerifiedPricingChecks -v'
} else {
 $dockerArgs+=@('--env-file',$envFile)
 $command='cp -a /source/. /app/ && exec python -B /trial/run_native.py '+$Mode
}
$dockerArgs+=@('sha256:e7847a00182ede9d93334b4f7cdd9688645c420865cddeba69feb3d4bc2ec219','sh','-c',$command)
$ErrorActionPreference='Continue'
& $dockerExe @dockerArgs 2>&1 | Tee-Object -FilePath (Join-Path $runDir ($Mode+'_console.log'))
$code=$LASTEXITCODE
$ErrorActionPreference='Stop'
if($Mode -eq 'execute') {
 $remaining=& $dockerExe ps -q --filter "name=^/$name`$"
 [ordered]@{exit_code=$code;container_name=$name;container_running=[bool]$remaining;local_process_tree_ended=(-not [bool]$remaining)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'CONTAINER_FINISHED.json') -Encoding utf8
}
exit $code
