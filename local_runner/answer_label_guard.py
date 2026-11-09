"""Evaluation-only boundary: withhold external answers to the frozen question.

No known answer, candidate, question ID or previous research is used, and no
site is named for one question: the source rule below covers kinds of host.
This detector is intentionally limited; post-run contamination audit is required.
A hit on a search entry also quarantines that source for the rest of the run,
because a later snippet of the same paper need not repeat the matched text.
"""
import hashlib, html, importlib, json, math, re, threading, time, unicodedata
from pathlib import Path

if __package__:
    from .tool_classification import is_internal_tool
else:
    from tool_classification import is_internal_tool

VERSION='question_source_guard_v6'
WINDOW_WORDS=8
# A real source can share one phrase with the question; a copy of the question
# shares much more. Observed longest runs: 7-16% for sources, 29-100% for copies.
MIN_QUESTION_SHARE=0.2
MARKER='[Evaluation source withheld: this external content reproduces part of the current question and may disclose a benchmark solution. Use independent factual sources.]'
SEEK_MARKER='[Evaluation request withheld: this run may not look for a benchmark or reference answer to the current question. Use independent factual sources.]'
# Strong labels only: a bare "a:" or "answer:" also occurs in ordinary pages.
LABEL=re.compile(r'\b(?:gold[ -]*(?:standard[ -]*)?answer|ground[ -]*truth|(?:original(?:ly)?[ -]*)?annotated[ -]*answer|reference[ -]*answer|expected[ -]*answer|correct[ -]*answer|revised[ -]*(?:answer|a)\s*:|candidate[ -]*answer\s*:|rationale\s*:|explanation\s*:)',re.I)
SEEKING=re.compile(r'\b(?:gold|correct|expected|reference|stated)[\s/()-]*(?:gold|correct|standard)?[\s/()-]*answers?\b|\bground[ -]*truth\b|\banswer[ -]*key\b|\bbenchmark[ -]*(?:answers?|questions?|dataset)\b|\bbrowsecomp\b',re.I)
# Hosted datasets and evaluation logs publish run transcripts whose snippets can
# stay under the question-share threshold, so these are matched by URL and title.
EVAL_SOURCE=re.compile(r'(?:huggingface\.co|kaggle\.com)/datasets/|·\s*Datasets at\b|\beval(?:uation)?[-_ ]logs?\b|\bsimple-evals\b|\bbrowsecomp\b',re.I)
ENTRY_TITLE=re.compile(r'^\[\d+\] \*\*(.+?)\*\*',re.M)
ENTRY_URL=re.compile(r'^\s*URL: (\S+)',re.M)
FETCH_WITHHELD=re.compile(r'^\[\d+\] URL: (\S+)\n\s+Info: '+re.escape(MARKER),re.M)
SEARCH_TOOLS=frozenset({'web_search','scholar_search'})

def words(text):
    text=unicodedata.normalize('NFKC',html.unescape(re.sub(r'<[^>]*>',' ',str(text)))).casefold()
    return re.findall(r'\w+',text)

def url_key(url):
    url=re.sub(r'^[a-z]+://(?:www\.)?','',str(url).strip().casefold())
    return 'url:'+url.split('#')[0].rstrip('/')

def entry_keys(entry):
    title=ENTRY_TITLE.search(entry);url=ENTRY_URL.search(entry)
    keys=set()
    if title:keys.add('title:'+' '.join(words(title.group(1))))
    if url:keys.add(url_key(url.group(1)))
    return keys

def requested_urls(args):
    value=args.get('url') if isinstance(args,dict) else None
    if isinstance(value,(list,tuple)):return [str(item) for item in value]
    return [str(value)] if value else []

class AnswerLabelGuard:
    def __init__(self,question,output):
        tokens=words(question)
        self.width=min(max(WINDOW_WORDS,math.ceil(len(tokens)*MIN_QUESTION_SHARE)),len(tokens))
        self.spans={tuple(tokens[i:i+self.width]) for i in range(len(tokens)-self.width+1)} if tokens else set()
        self.qhash=hashlib.sha256(question.encode()).hexdigest()
        self.output=Path(output);self.lock=threading.Lock()
        self.quarantine=set();self.recorded=set()

    def is_answer_source(self,text):
        tokens=words(text)
        return bool(self.spans) and any(tuple(tokens[i:i+self.width]) in self.spans for i in range(len(tokens)-self.width+1))

    def seeks_answer(self,value):
        text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,default=str)
        return bool(SEEKING.search(text))

    def record(self,text,stage):
        digest=hashlib.sha256(text.encode()).hexdigest()
        with self.lock:
            # The observer and the post-processor screen the same result.
            if digest in self.recorded:return
            self.recorded.add(digest)
            folder=self.output/'guard_withheld';folder.mkdir(parents=True,exist_ok=True)
            p=folder/(digest+'.txt')
            if not p.exists():p.write_text(text,encoding='utf-8')
            with (self.output/'ANSWER_LABEL_GUARD.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'version':VERSION,'ts':time.time(),'stage':stage,'sha256':digest,
                    'chars':len(text),'question_sha256':self.qhash,'action':'withheld_before_model_delivery'})+'\n')

    def withhold_request(self,name,value):
        text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,default=str)
        self.record(text,'answer_seeking_request:'+str(name))
        return SEEK_MARKER

    def filter_entry(self,entry):
        keys=entry_keys(entry)
        quarantined=bool(keys&self.quarantine)
        title,url=ENTRY_TITLE.search(entry),ENTRY_URL.search(entry)
        evaluation=bool(EVAL_SOURCE.search(' '.join(m.group(1) for m in (title,url) if m)))
        if not (quarantined or evaluation or self.is_answer_source(entry) or LABEL.search(entry)):return entry
        self.quarantine|=keys
        self.record(entry,'search_quarantined_source' if quarantined else 'search_eval_source' if evaluation else 'search_source_entry')
        return MARKER+'\n\n'

    def filter_search(self,text):
        # Query text is generated by the agent, not evidence from a source.
        # Preserve other search hits when one entry reproduces the benchmark.
        chunks=re.split(r'(?m)(?=^\[\d+\] \*\*|^## Query:)',text)
        out=[]
        for chunk in chunks:
            if chunk.startswith('## Query:'):
                head,sep,body=chunk.partition('\n')
                out.append(head+sep+self.filter(body,'search_nonentry_source'))
            else:out.append(self.filter_entry(chunk))
        return ''.join(out)

    def filter_fetch(self,text,args,stage):
        urls=requested_urls(args)
        if any(url_key(url) in self.quarantine or EVAL_SOURCE.search(url) for url in urls):
            self.record(text,'fetch_quarantined_source')
            return MARKER
        filtered=self.filter(text,stage)
        if filtered==MARKER:self.quarantine|={url_key(url) for url in urls}
        self.quarantine|={url_key(url) for url in FETCH_WITHHELD.findall(filtered)}
        return filtered

    def filter(self,text,stage):
        if not isinstance(text,str) or not self.is_answer_source(text):return text
        self.record(text,stage)
        return MARKER

    def screen(self,name,text,args=None):
        if args and self.seeks_answer(args):return self.withhold_request(name,args)
        return self.filter_search(text) if name in SEARCH_TOOLS else self.filter_fetch(text,args,'tool_result:'+name)

def install_guard(question,output):
    from workflows.agent_team import convergence, subagent_runtime
    web_fetch_aligned=importlib.import_module('plugins.tools.web_fetch_aligned')
    web_fetch=importlib.import_module('plugins.tools.web_fetch')
    from frontier_agent.infra import summary_llm
    guard=AnswerLabelGuard(question,output)
    original_process=subagent_runtime.SwarmToolResultPostProcessor.process
    original_observe=convergence.ConvergenceState.observe_main_tool
    original_aligned=web_fetch_aligned._extract_info_with_llm
    original_summary=summary_llm.summarize
    original_fetch_alias=web_fetch._summary_llm_summarize

    def process(self,result):
        name=result.name or ''
        if not is_internal_tool(name):
            text=result.result if isinstance(result.result,str) else str(result.result)
            filtered=guard.screen(name,text,getattr(result,'args',None))
            if filtered!=text:return filtered
        return original_process(self,result)

    def observe(self,*,tool_call_id,tool_name,arguments,result,is_error=False,error_kind=''):
        # Observers run before the post-processor. Receipts reach helper
        # prompts and the team handoff, so they must hold the screened text.
        if not is_internal_tool(tool_name):
            text=result if isinstance(result,str) else str(result)
            result=guard.screen(tool_name,text,arguments)
        return original_observe(self,tool_call_id=tool_call_id,tool_name=tool_name,
            arguments=arguments,result=result,is_error=is_error,error_kind=error_kind)

    async def aligned(content,info_to_extract,truncate_last_num_chars=-1):
        if guard.seeks_answer(info_to_extract):
            return {'success':True,'extracted_info':guard.withhold_request('web_fetch',info_to_extract),'error':''}
        filtered=guard.filter(content,'aligned_fetch_before_summary')
        if filtered!=content:return {'success':True,'extracted_info':filtered,'error':''}
        return await original_aligned(content,info_to_extract,truncate_last_num_chars)

    async def summary(content,focus):
        filtered=guard.filter(content,'web_fetch_before_summary')
        if filtered!=content:return filtered
        return await original_summary(content,focus)

    subagent_runtime.SwarmToolResultPostProcessor.process=process
    convergence.ConvergenceState.observe_main_tool=observe
    web_fetch_aligned._extract_info_with_llm=aligned
    summary_llm.summarize=summary
    web_fetch._summary_llm_summarize=summary
    def restore():
        subagent_runtime.SwarmToolResultPostProcessor.process=original_process
        convergence.ConvergenceState.observe_main_tool=original_observe
        web_fetch_aligned._extract_info_with_llm=original_aligned
        summary_llm.summarize=original_summary
        web_fetch._summary_llm_summarize=original_fetch_alias
    return guard,restore
