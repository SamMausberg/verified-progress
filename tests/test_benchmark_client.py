"""Local protocol tests only. These are not model or GPU benchmarks."""
import sys,json,unittest,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_sse import events,run,one_request
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        obj=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        messages=[{'choices':[{'delta':{'content':'several tokens in one chunk'},'finish_reason':None}]}]
        if obj['model']!='missing_usage':messages.append({'choices':[],'usage':{'completion_tokens':7}})
        for m in messages:self.wfile.write(('data: '+json.dumps(m)+'\n\n').encode());self.wfile.flush()
        if obj['model']!='truncated':self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}/v1/chat/completions'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def test_sse_multiline(self):
        self.assertEqual(list(events([b':comment\n',b'data: first\n',b'data: second\n',b'\n'])),['first\nsecond'])
    def test_usage_not_chunks(self):
        s,rows=run(self.url,'test',[{'id':i,'prompt':'test','max_tokens':8} for i in range(8)],2)
        self.assertEqual(s['completed_output_tokens'],56);self.assertTrue(s['valid_comparison'])
        self.assertTrue(all(len(x['chunk_offsets_s'])==1 for x in rows))
    def test_missing_usage_fails(self):
        x=one_request(self.url,'missing_usage',{'prompt':'x'},0,5)
        self.assertFalse(x['success']);self.assertIn('refusing to count chunks',x['error'])
    def test_truncated_stream_fails(self):
        x=one_request(self.url,'truncated',{'prompt':'x'},0,5)
        self.assertFalse(x['success']);self.assertIn('without terminal',x['error'])
if __name__=='__main__':unittest.main(verbosity=2)
