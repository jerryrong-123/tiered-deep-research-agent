"""Offline checks of the actual provider/stream boundary and native workflow."""
import asyncio, json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import httpx
import run_native as r

class Checks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.old_here=r.HERE;self.old_helper_here=r.helpers.HERE
        r.HERE=Path(self.tmp.name);r.helpers.HERE=r.HERE
        self.old_send=httpx.AsyncClient.send
    def tearDown(self):
        httpx.AsyncClient.send=self.old_send;r.HERE=self.old_here;r.helpers.HERE=self.old_helper_here;self.tmp.cleanup()
    def test_limits_and_reasoning_roundtrip(self):
        p={'model':'deepseek-flash','max_completion_tokens':32768,'messages':[{'role':'assistant','reasoning_content':'private-memory-only','tool_calls':[]}],
            'thinking':{'type':'enabled'},'reasoning_effort':'high'}
        v=r.provider_payload(p)
        self.assertEqual(v['max_tokens'],32768);self.assertNotIn('max_completion_tokens',v)
        self.assertEqual(v['messages'],p['messages']);self.assertEqual(v['reasoning_effort'],'high')
        self.assertNotIn('private-memory-only',json.dumps(r.public(v)))
        with self.assertRaises(RuntimeError):r.provider_payload({**p,'max_tokens':10})
    async def test_stream_chunk_boundaries_usage_and_public_text(self):
        meter=r.Meter();idx=await meter.reserve({'max_tokens':100},'offline')
        frames=[{'choices':[{'index':0,'delta':{'reasoning_content':'private-memory-only'}}]},
            {'model':'deepseek-flash','choices':[{'index':0,'delta':{'content':'public answer'},'finish_reason':'stop'}]},
            {'choices':[],'usage':{'prompt_tokens':30,'completion_tokens':8,'prompt_cache_hit_tokens':10}}]
        raw=b''.join(b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames)+b'data: [DONE]\n\n'
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for i in range(0,len(raw),7):yield raw[i:i+7]
        stream=r.RecordedStream(Stream(),idx,meter,True,200)
        returned=b''.join([c async for c in stream]);await stream.aclose()
        self.assertEqual(returned,raw);self.assertEqual(meter.held,0);self.assertEqual(meter.reserved,0)
        saved=(r.HERE/'responses/00000.json').read_text()
        self.assertNotIn('private-memory-only',saved);self.assertIn('public answer',saved)
    async def test_incomplete_response_retains_unknown(self):
        meter=r.Meter();idx=await meter.reserve({'max_tokens':100},'offline')
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[]}\n\n'
                raise httpx.RemoteProtocolError('peer closed connection without sending complete message body')
        stream=r.RecordedStream(Stream(),idx,meter,True,200)
        with self.assertRaises(httpx.RemoteProtocolError):
            async for _ in stream:pass
        self.assertGreater(meter.held,0);self.assertEqual(meter.reserved,0)
    async def test_native_pipeline_and_provider_without_network(self):
        r.credentials();os.chdir(r.UP)
        meter=r.Meter();r.install_transport(meter,offline=True)
        requests=[]
        async def responder(request):
            payload=json.loads(request.content);requests.append(payload)
            self.assertEqual(payload['model'],'deepseek-flash');self.assertIn('max_tokens',payload)
            self.assertNotIn('max_completion_tokens',payload)
            if not payload.get('stream'):
                return httpx.Response(200,json={'id':'offline','object':'chat.completion','created':0,'model':'deepseek-flash',
                    'choices':[{'index':0,'message':{'role':'assistant','content':'Offline interface smoke test completed.'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'prompt_cache_hit_tokens':0}})
            frames=[{'id':'offline','object':'chat.completion.chunk','created':0,'model':'deepseek-flash',
                'choices':[{'index':0,'delta':{'role':'assistant','content':'Offline interface smoke test completed.'},'finish_reason':None}]},
                {'id':'offline','object':'chat.completion.chunk','created':0,'model':'deepseek-flash',
                'choices':[{'index':0,'delta':{},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'prompt_cache_hit_tokens':0}}]
            data=b''.join(b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames)+b'data: [DONE]\n\n'
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content=data)
        original_init=httpx.AsyncClient.__init__
        def init(client,*a,**kw):
            kw['transport']=httpx.MockTransport(responder);kw['trust_env']=False;original_init(client,*a,**kw)
        with patch.object(httpx.AsyncClient,'__init__',init):
            from benchmarks.public.core.kernel_adapter import BenchmarkSession
            from workflows.agent_team.nodes.main_agent import _llm_cache
            _llm_cache.clear()
            async with BenchmarkSession() as session:
                result=await session.run('Return a short offline interface smoke-test response.',pipeline_id='agent_team',
                    meta={'profile':'benchmark','profile_inline':r.selected_profile(),'profile_overrides':r.overrides(),'_trial_dir':str(r.HERE/'native')})
        self.assertTrue(requests)
        # This mock answers with no research. A light-stage run must refuse that
        # and stop with its marker; every other profile delivers the mock text.
        controls=r.selected_profile()['agent'].get('progressive_config') or {}
        expected='[ESCALATION_REQUIRED]' if controls.get('stop_at_escalation') else 'Offline interface'
        self.assertIn(expected,str(result.get('final_answer') or result.get('final_content')))
        if controls.get('synthesize_at_stop'):
            # The closing call must clear the budget layer and reach the provider.
            self.assertIn('research budget for this question has ended',json.dumps(requests[-1]['messages']))
            saved=next((r.HERE/'native').rglob('convergence_v1.json')).read_text(encoding='utf-8')
            self.assertNotIn('closing_synthesis_failed',saved);self.assertIn('answer_settled',saved)
        r.save(r.HERE/'STATE.json',r.public(result))
        self.assertEqual(meter.held,0)

if __name__=='__main__':unittest.main(verbosity=2)
