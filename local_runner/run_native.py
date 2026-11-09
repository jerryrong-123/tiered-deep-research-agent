"""One native Agent Team trial. Instrument transport; retain upstream workflow."""
from __future__ import annotations
import asyncio, contextlib, hashlib, importlib.util, json, logging, os, sys, time
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
import httpx

HERE=Path(__file__).resolve().parent
ROOT=HERE
UP=Path('/app')
DEADLINE=14531.469
if __package__:
    from . import trial_helpers as helpers
    from .budget_adapter import make_meter_class, safe_error_with_budget
    from .sandbox_registry import SandboxRegistry, read_sandbox_events
else:
    import trial_helpers as helpers
    from budget_adapter import make_meter_class, safe_error_with_budget
    from sandbox_registry import SandboxRegistry, read_sandbox_events
helpers.HERE=HERE
def public(value):
    if isinstance(value,dict):
        return {str(k):public(v) for k,v in value.items() if str(k).casefold() not in helpers.HIDDEN and not (str(k).casefold()=='thinking' and isinstance(v,str))}
    if isinstance(value,(list,tuple)):return [public(v) for v in value]
    return value
helpers.public=public
save=helpers.save; emit=helpers.emit; safe_error=safe_error_with_budget(helpers.safe_error)
sys.path.insert(0,str(UP))

def credentials():
    for name in ('DEEPSEEK_API_KEY','JINA_API_KEY','SERPER_API_KEY','E2B_API_KEY'):
        if not os.environ.get(name):raise RuntimeError('missing_'+name)
    assert os.environ['OPENAI_MODEL'] in helpers.RATES
    assert os.environ['OPENAI_BASE_URL']=='https://api.deepseek.com'
    assert os.environ['SANDBOX_BACKEND']=='e2b'
    if os.environ['OPENAI_MODEL']!='deepseek-flash':
        # Page extraction stays on the cheap model whichever model reasons.
        os.environ.update(SUMMARY_LLM_MODEL_NAME='deepseek-flash',
            SUMMARY_LLM_BASE_URL='https://api.deepseek.com/chat/completions',
            SUMMARY_LLM_API_KEY=os.environ['DEEPSEEK_API_KEY'])
    return None

def check_source():
    baseline=json.loads((HERE/'IMAGE_SOURCE_BASELINE.json').read_text())
    for name,digest in baseline['files'].items():
        if hashlib.sha256((UP/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('source_changed:'+name)
    q=json.loads((HERE/'QUESTION.json').read_text(encoding='utf-8'))
    assert set(q)=={'id','question','question_sha256'}
    assert hashlib.sha256(q['question'].encode()).hexdigest()==q['question_sha256']
    return q

def selected_profile():
    import yaml
    return yaml.safe_load((Path(__file__).resolve().parent/'PROFILE.yaml').read_text(encoding='utf-8'))

def overrides():
    return {'llm':{'extra_body':{'top_k':None,'thinking':{'type':'enabled'},'reasoning_effort':'high'}},
        'agent':{'thinking_format':'reasoning_content','wall_deadline_s':DEADLINE,'reporter':False}}

Meter = make_meter_class(helpers)

class RecordOnlyMeter(Meter):
    def snapshot(self):
        data=super().snapshot()
        data.update(target_cny=None,hard_cap_cny=None,budget_mode='record_only_user_authorized',scope='fresh_attempt_only')
        return data
    async def reserve(self,payload,role):
        amount=(Decimal(1048576)*2+Decimal(payload['max_tokens'])*8)/Decimal(1000000)
        async with self.lock:
            ident=self.calls;self.calls+=1;self.reserved+=amount
            self.entries[ident]={'amount':amount,'started':datetime.now(timezone.utc),'input_upper':1048576}
            self.write();emit('model_admitted',call=ident,role=role,reserved_upper_cny=str(amount),budget_mode='record_only')
        return ident

def provider_payload(data):
    data=dict(data)
    if data.get('model') not in helpers.RATES: raise RuntimeError('unexpected_model')
    if 'max_completion_tokens' in data:
        value=data.pop('max_completion_tokens')
        if 'max_tokens' in data and data['max_tokens']!=value: raise RuntimeError('conflicting_output_limits')
        data['max_tokens']=value
    if not isinstance(data.get('max_tokens'),int) or not 0<data['max_tokens']<=1048576:
        raise RuntimeError('missing_or_invalid_output_cap')
    return data

class RecordedStream(httpx.AsyncByteStream):
    def __init__(self,inner,ident,meter,is_sse,status):
        self.inner=inner;self.ident=ident;self.meter=meter;self.is_sse=is_sse;self.status=status
        self.buffer=b'';self.usage=None;self.closed=False;self.complete=False;self.done=False
        self.bytes=0;self.chunks=0;self.started=time.monotonic();self.last=0.;self.first=False
        self.content={};self.tools={};self.finish={};self.reasoning_chars=0;self.model=None
    def frame(self,line):
        if not line.startswith(b'data:'): return
        raw=line[5:].strip()
        if raw==b'[DONE]': self.done=True;return
        if not raw:return
        try:data=json.loads(raw)
        except ValueError:return
        if data.get('usage'):self.usage=data['usage']
        self.model=data.get('model',self.model)
        for c in data.get('choices',[]):
            idx=c.get('index',0);delta=c.get('delta') or {}
            self.reasoning_chars+=len(delta.get('reasoning_content') or '')
            if delta.get('content'):self.content[idx]=self.content.get(idx,'')+delta['content']
            if c.get('finish_reason'):self.finish[idx]=c['finish_reason']
            for t in delta.get('tool_calls') or []:
                key=(idx,t.get('index',0));dst=self.tools.setdefault(key,{'id':'','type':'function','function':{'name':'','arguments':''}})
                if t.get('id'):dst['id']=t['id']
                for k in ('name','arguments'):
                    if t.get('function',{}).get(k):dst['function'][k]+=t['function'][k]
    async def __aiter__(self):
        try:
            async for chunk in self.inner:
                self.bytes+=len(chunk);self.chunks+=1
                if not self.first:self.first=True;emit('first_body_bytes',call=self.ident,elapsed_s=time.monotonic()-self.started)
                self.buffer+=chunk
                if self.is_sse:
                    while b'\n' in self.buffer:
                        line,self.buffer=self.buffer.split(b'\n',1);self.frame(line.strip())
                now=time.monotonic()
                if now-self.last>=20:
                    self.last=now;emit('model_stream_progress',call=self.ident,bytes=self.bytes,chunks=self.chunks,
                        public_chars=sum(map(len,self.content.values())),reasoning_chars=self.reasoning_chars,tool_calls=len(self.tools))
                yield chunk
            self.complete=True
        except BaseException as exc:
            emit('model_body_failed',call=self.ident,**safe_error(exc));raise
        finally:
            await self.finish_account()
    async def finish_account(self):
        if self.closed:return
        self.closed=True
        if not self.is_sse:
            try:
                data=json.loads(self.buffer);self.usage=data.get('usage')
                save(HERE/'responses'/f'{self.ident:05d}.json',public(data))
                self.model=data.get('model');self.finish={i:c.get('finish_reason') for i,c in enumerate(data.get('choices',[]))}
            except (ValueError,AttributeError):pass
        else:
            self.frame(self.buffer.strip())
            indexes=set(self.content)|set(self.finish)|{k[0] for k in self.tools}
            save(HERE/'responses'/f'{self.ident:05d}.json',{'scope':'public_stream_projection','model':self.model,
                'choices':[{'index':i,'message':{'role':'assistant','content':self.content.get(i,''),
                'tool_calls':[t for (ci,_),t in self.tools.items() if ci==i]},'finish_reason':self.finish.get(i)} for i in sorted(indexes)],
                'usage':self.usage,'body_complete':self.complete,'done_marker':self.done,'reasoning_chars':self.reasoning_chars})
        await self.meter.settle(self.ident,self.usage)
        emit('model_response_finished',call=self.ident,status=self.status,body_complete=self.complete,done_marker=self.done,
            finish_reasons=self.finish,bytes=self.bytes,chunks=self.chunks,usage=helpers.numeric_usage(self.usage),
            elapsed_s=time.monotonic()-self.started)
    async def aclose(self):
        try:await self.inner.aclose()
        finally:await self.finish_account()


def cleanup_sandboxes():
    registry=SandboxRegistry(HERE)
    records,invalid_registry=registry.records()
    ids,invalid_log=read_sandbox_events(HERE/'events.jsonl')
    ids.update(ident for ident,row in records.items() if row['status']=='active')
    ids.difference_update(ident for ident,row in records.items() if row['status']=='cleanup_confirmed')
    summary={'version':'sandbox_cleanup_v1','invalid_log_lines':invalid_log,
             'invalid_registry_records':invalid_registry,'checks':[],'event_write_errors':[]}
    def record(kind,**values):
        try:emit(kind,**values)
        except Exception as exc:summary['event_write_errors'].append(safe_error(exc))
    for issue in invalid_log:record('sandbox_cleanup_log_invalid',**issue)
    for issue in invalid_registry:record('sandbox_cleanup_registry_invalid',**issue)
    for ident in sorted(ids):
        try:
            from e2b_code_interpreter import Sandbox
            killed=Sandbox.kill(ident,api_key=os.environ['E2B_API_KEY'])
        except Exception as exc:check={'sandbox_id':ident,'confirmed':False,**safe_error(exc)}
        else:
            check={'sandbox_id':ident,'confirmed':True,'already_absent':not killed}
            try:registry.confirm_cleanup(ident)
            except Exception as exc:check['registry_write_error']=safe_error(exc)
        summary['checks'].append(check)
        record('sandbox_cleanup',**check)
    save(HERE/'SANDBOX_CLEANUP.json',summary)
    return summary


def register_created_sandbox(sandbox):
    """Persist the actual SDK id before exposing a created sandbox or logging it."""
    ident=getattr(sandbox,'sandbox_id',None)
    try:SandboxRegistry(HERE).register(ident)
    except BaseException:
        # A successfully returned remote resource must not escape unregistered.
        # If persistence fails, immediately try to close this exact SDK object.
        try:sandbox.kill()
        except Exception as exc:
            # Preserve the known id in the legacy fallback if the independent
            # registry is unavailable and the immediate close also failed.
            if isinstance(ident,str) and ident:
                with contextlib.suppress(Exception):
                    emit('sandbox_created',sandbox_id=ident,registry_persisted=False,
                         registration_cleanup_failed=True,**safe_error(exc))
        raise
    emit('sandbox_created',sandbox_id=ident)
    return sandbox

class ReaderRecordedStream(httpx.AsyncByteStream):
    def __init__(self,inner,request,status,headers,host='r.jina.ai'):
        # A pinned read dials an address, so the URL has to carry the site's name again.
        self.inner=inner;self.host=host;self.url=str(request.url.copy_with(host=host));self.status=status;self.headers=dict(headers);self.body=bytearray();self.complete=False;self.saved=False
    async def __aiter__(self):
        try:
            async for chunk in self.inner:
                self.body.extend(chunk);yield chunk
            self.complete=True
        finally:self.persist()
    def persist(self):
        if self.saved:return
        self.saved=True
        digest=hashlib.sha256(self.body).hexdigest()
        record={'host':self.host,'url':self.url,'status':self.status,'sha256':digest,'body_complete':self.complete,
            'content_encoding':self.headers.get('content-encoding','identity')}
        try:record['text']=httpx.Response(self.status,headers=self.headers,content=bytes(self.body)).text
        except Exception:
            import base64
            record['wire_body_base64']=base64.b64encode(self.body).decode();record['decode_status']='incomplete_or_unsupported_encoding'
        save(HERE/'web_responses'/(digest+'.json'),record)
    async def aclose(self):
        try:await self.inner.aclose()
        finally:self.persist()

def install_transport(meter,offline=False):
    original=httpx.AsyncClient.send
    async def send(client,request,**kwargs):
        host=request.url.host
        if host!='api.deepseek.com' or not request.url.path.endswith('/chat/completions'):
            if offline: raise RuntimeError('offline_network_blocked')
            # Web observations are retained in native tool trajectories and separately here.
            # Saved pages are read directly, and the source audit has to see them too.
            site=request.headers.get('host',host)
            is_web=site in {'google.serper.dev','r.jina.ai','s.jina.ai','archive.org','web.archive.org'}
            if is_web:emit('web_request',host=site,method=request.method,path=request.url.path)
            response=await original(client,request,**kwargs)
            if is_web:
                emit('web_response',host=site,status=response.status_code)
                if site in {'r.jina.ai','web.archive.org'} and kwargs.get('stream'):
                    response.stream=ReaderRecordedStream(response.stream,request,response.status_code,response.headers,site)
                if not kwargs.get('stream'):
                    ident=hashlib.sha256(response.content).hexdigest()
                    save(HERE/'web_responses'/(ident+'.json'),{'host':site,'url':str(request.url),
                        'status':response.status_code,'text':response.text,'sha256':ident})
            return response
        data=provider_payload(json.loads(request.content))
        ident=await meter.reserve(data,'native_stream' if data.get('stream') else 'native_completion')
        try:
            save(HERE/'requests'/f'{ident:05d}.json',{'scope':'public_projection','request':public(data),
                'full_request_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()})
            emit('model_request',call=ident,model=data['model'],max_tokens=data['max_tokens'],
                stream=data.get('stream',False),thinking=data.get('thinking'),reasoning_effort=data.get('reasoning_effort'),
                tools=[t.get('function',{}).get('name') for t in data.get('tools',[])])
            headers=dict(request.headers);headers.pop('content-length',None);headers['accept-encoding']='identity'
            extensions=dict(request.extensions)
            async def trace(name,info):emit('http_phase',call=ident,phase=name)
            extensions['trace']=trace
            req=httpx.Request(request.method,request.url,headers=headers,content=json.dumps(data).encode(),extensions=extensions)
            requested_stream=kwargs.pop('stream',False)
        except BaseException as exc:
            await meter.cancel(ident);emit('model_request_setup_failed',call=ident,**safe_error(exc));raise
        try:
            await meter.mark_dispatched(ident)
            response=await original(client,req,stream=True,**kwargs)
        except BaseException as exc:
            await meter.cancel(ident);emit('model_transport_failed',call=ident,**safe_error(exc));raise
        emit('http_headers',call=ident,status=response.status_code,headers={k:response.headers[k] for k in
            ('content-type','content-length','transfer-encoding','content-encoding','x-request-id','x-ds-request-id') if k in response.headers})
        response.stream=RecordedStream(response.stream,ident,meter,'text/event-stream' in response.headers.get('content-type',''),response.status_code)
        if not requested_stream:await response.aread()
        if response.status_code in (401,402):meter.reason='model_auth_or_funding_unavailable';meter.stop.set()
        return response
    httpx.AsyncClient.send=send
    return original

def observation_hooks():
    from agent_core.components.observers.trajectory import TrajectoryFileObserver
    originals={}
    for name in ('_write_jsonl','_append_message'):
        old=getattr(TrajectoryFileObserver,name)
        originals[name]=old
        def wrapped(self,record,_old=old):return _old(self,public(record))
        setattr(TrajectoryFileObserver,name,wrapped)
    from e2b_code_interpreter import Sandbox
    import inspect
    create_descriptor=inspect.getattr_static(Sandbox,'create')
    create=Sandbox.create
    def recorded_create(*args,**kwargs):
        return register_created_sandbox(create(*args,**kwargs))
    Sandbox.create=staticmethod(recorded_create)
    # Log public tool events from all native loops without changing their behavior.
    old=TrajectoryFileObserver.on_tool_result
    async def tool_result(self,ctx,result):
        emit('tool_result',role=self._role_id,task=self._task_id,tool=getattr(result,'tool_name',''),
            success=getattr(result,'success',None))
        return await old(self,ctx,result)
    TrajectoryFileObserver.on_tool_result=tool_result
    def restore():
        for name,method in originals.items():setattr(TrajectoryFileObserver,name,method)
        TrajectoryFileObserver.on_tool_result=old
        Sandbox.create=create_descriptor
    return restore

def safe_logging():
    keys=[os.environ.get(n,'') for n in ('DEEPSEEK_API_KEY','SERPER_API_KEY','JINA_API_KEY','E2B_API_KEY')]
    class Filter(logging.Filter):
        def filter(self,record):
            text=record.getMessage()
            for key in keys:
                if key:text=text.replace(key,'[REDACTED]')
            record.msg=text;record.args=();return True
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    for h in logging.getLogger().handlers:h.addFilter(Filter())

async def worker():
    if json.loads(sys.stdin.read()).get('start') is not True:raise RuntimeError('supervisor_handshake_missing')
    credentials();q=check_source();safe_logging();os.chdir(UP)
    meter=Meter();meter.write();install_transport(meter);observation_hooks()
    if __package__:
        from .answer_label_guard import install_guard
    else:
        from answer_label_guard import install_guard
    install_guard(q['question'],HERE)
    from benchmarks.public.core.kernel_adapter import BenchmarkSession
    from workflows.agent_team.profile import load_swarm_profile
    from frontier_agent.infra.usage_meter import ExternalAPIMeter, bind_usage_meter, reset_usage_meter
    external=ExternalAPIMeter();external_token=bind_usage_meter(external)
    async def snapshot_usage():
        while True:
            save(HERE/'EXTERNAL_API_USAGE.json',external.snapshot())
            await asyncio.sleep(10)
    usage_task=asyncio.create_task(snapshot_usage())
    profile=load_swarm_profile('benchmark',overrides=overrides(),inline=selected_profile())
    meter.configure_final_output_upper(profile['llm']['max_tokens'],os.environ['OPENAI_MODEL'])
    save(HERE/'EFFECTIVE_PROFILE.json',public(profile))
    result={'case':q['id'],'attempt':1,'model':os.environ['OPENAI_MODEL'],'automatic_judge_calls':0,'gold_read':False}
    start=time.monotonic();pipeline=None;stopper=None
    try:
        async with BenchmarkSession() as session:
            pipeline=asyncio.create_task(session.run(q['question'],pipeline_id='agent_team',meta={
                'profile':'benchmark','profile_inline':selected_profile(),'profile_overrides':overrides(),'_trial_dir':str(HERE/'native'),
                'question_id':q['id'],'pipeline_id':'agent_team','benchmark':'browsecomp'}))
            stopper=asyncio.create_task(meter.stop.wait())
            done,_=await asyncio.wait([pipeline,stopper],timeout=DEADLINE,return_when=asyncio.FIRST_COMPLETED)
            if pipeline in done:
                state=pipeline.result();save(HERE/'STATE.json',public(state))
                result.update(technical_status='pipeline_returned',answer=state.get('final_answer') or state.get('final_content'),
                    stopped_by=state.get('stopped_by'),answer_status=state.get('answer_status'),errors=public(state.get('errors')),correct=None)
            else:
                pipeline.cancel()
                with contextlib.suppress(BaseException):await pipeline
                result.update(technical_status='technical_incomplete',reason=meter.reason or 'whole_question_deadline')
    except BaseException as exc:
        import traceback
        result.update(technical_status='technical_incomplete',reason=meter.reason or 'runtime_failure',**safe_error(exc))
        result['traceback_frames']=[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
        emit('runtime_failed',**safe_error(exc))
    finally:
        meter.release_final_allowance()
        usage_task.cancel()
        if stopper:stopper.cancel()
        from plugins.tools._sandbox import close_sandbox
        await asyncio.to_thread(close_sandbox)
        save(HERE/'EXTERNAL_API_USAGE.json',external.snapshot());reset_usage_meter(external_token)
        result.update(duration_s=time.monotonic()-start,model_ledger=meter.snapshot())
        save(HERE/'WORKER_RESULT.json',result)
        check_source()

def supervisor():
    credentials();check_source()
    save(HERE/'MODEL_BALANCE_BEFORE.json',helpers.balance())
    with (HERE/'STARTED.json').open('x',encoding='utf-8') as f:json.dump({'utc':helpers.stamp(),'attempt':1},f)
    sys.path.insert(0,str(ROOT));from task_supervisor import supervise
    status=supervise([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'worker'],{'start':True},
        timeout_s=DEADLINE,log_path=HERE/'worker_console.log',cwd=UP)
    save(HERE/'SUPERVISOR.json',status)
    cleanup_sandboxes()
    save(HERE/'MODEL_BALANCE_AFTER.json',helpers.balance())
    lp=HERE/'MODEL_LEDGER.json'
    if lp.exists():
        ledger=json.loads(lp.read_text());remaining=Decimal(ledger['inflight_reserved_upper_cny'])
        if remaining:
            save(HERE/'LEDGER_BEFORE_SUPERVISOR_SETTLEMENT.json',ledger)
            ledger['unknown_held_upper_cny']=str(Decimal(ledger['unknown_held_upper_cny'])+remaining)
            ledger['inflight_reserved_upper_cny']='0';save(lp,ledger)
    save(HERE/'FINISHED.json',{'utc':helpers.stamp(),'supervisor':status,'worker_result_exists':(HERE/'WORKER_RESULT.json').exists()})
    print(json.dumps(status))

if __name__=='__main__':
    if sys.argv[-1]=='worker':asyncio.run(worker())
    elif sys.argv[-1]=='execute':supervisor()
    elif sys.argv[-1]=='cleanup':credentials();cleanup_sandboxes()
    else:raise SystemExit('Explicit execute, worker or cleanup required')
