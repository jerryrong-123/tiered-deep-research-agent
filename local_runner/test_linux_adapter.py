import gzip
import json
from decimal import Decimal
import httpx
from test_native import Checks
import run_native as r

class LinuxChecks(Checks):
    def test_thinking_string_not_persisted_but_config_retained(self):
        data={'thinking':'PRIVATE_TEXT','config':{'thinking':{'type':'enabled'}}}
        value=r.public(data)
        self.assertNotIn('PRIVATE_TEXT',json.dumps(value))
        self.assertEqual(value['config']['thinking'],{'type':'enabled'})

    async def test_hard_budget_stops_without_network(self):
        meter=r.Meter();meter.spent=Decimal('25')
        with self.assertRaisesRegex(RuntimeError,'model_budget_admission_exhausted'):
            await meter.reserve({'max_tokens':32768},'offline')
        self.assertTrue(meter.stop.is_set());self.assertEqual(meter.calls,0)

    async def test_reader_gzip_preserves_public_body(self):
        raw=gzip.compress(b'Example Domain public reader body')
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for i in range(0,len(raw),5):yield raw[i:i+5]
        stream=r.ReaderRecordedStream(Stream(),httpx.Request('GET','https://r.jina.ai/https://example.com'),200,
                                      {'content-encoding':'gzip','content-type':'text/plain; charset=utf-8'})
        self.assertEqual(b''.join([c async for c in stream]),raw)
        await stream.aclose()
        saved=json.loads(next((r.HERE/'web_responses').glob('*.json')).read_text())
        self.assertEqual(saved['text'],'Example Domain public reader body')
        self.assertTrue(saved['body_complete'])

    async def test_a_pinned_saved_page_is_recorded_under_its_site(self):
        for old in (r.HERE/'web_responses').glob('*.json'):old.unlink()
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):yield b'<html>saved page</html>'
        request=httpx.Request('GET','https://207.241.237.3/web/20260730060433id_/https://example.com/',headers={'Host':'web.archive.org'})
        stream=r.ReaderRecordedStream(Stream(),request,200,{'content-type':'text/html'},request.headers['host'])
        self.assertEqual(b''.join([c async for c in stream]),b'<html>saved page</html>')
        await stream.aclose()
        saved=json.loads(next((r.HERE/'web_responses').glob('*.json')).read_text())
        self.assertEqual(saved['host'],'web.archive.org')
        self.assertEqual(saved['url'],'https://web.archive.org/web/20260730060433id_/https://example.com/')
        self.assertEqual(saved['text'],'<html>saved page</html>')

if __name__=='__main__':
    import unittest
    unittest.main(verbosity=2)
