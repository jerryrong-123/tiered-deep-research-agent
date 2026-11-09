"""Evaluation-only boundary: withhold external answers to the frozen question.

No known answer, candidate, question ID or previous research is used, and no
site is named for one question: the source rule below covers kinds of host.
This detector is intentionally limited; post-run contamination audit is required.
A hit on a search entry also quarantines that source for the rest of the run,
because a later snippet of the same paper need not repeat the matched text.

Since v8 the guard withholds whatever the post-run audit's A/B/C rules flag
(audit_abc.py, local_abc_v1_20261031), shell commands and downloads included.
A clean audit then means no flagged view reached a model, not that none existed.
"""
import hashlib, html, importlib, json, re, threading, time, unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

if __package__:
    from .tool_classification import is_internal_tool
else:
    from tool_classification import is_internal_tool

VERSION='question_source_guard_v8'
# The audit flags a view whose longest run of question words reaches 15% of the
# question (its gray band). Real sources reach 7-16%, so some are withheld too.
QUESTION_SHARE=(3,20)
# Neither marker may carry an audit label: a withheld view is still delivered.
MARKER='[Evaluation source withheld: this external content matches an evaluation-source rule for the current question. Use independent factual sources.]'
SEEK_MARKER='[Evaluation request withheld: this run may not look up a benchmark solution for the current question. Use independent factual sources.]'
# Strong labels only: a bare "a:" or "answer:" also occurs in ordinary pages.
LABEL=re.compile(r'\b(?:gold[ -]*(?:standard[ -]*)?answer|ground[ -]*truth|(?:original(?:ly)?[ -]*)?annotated[ -]*answer|reference[ -]*answer|expected[ -]*answer|correct[ -]*answer|revised[ -]*(?:answer|a)\s*:|candidate[ -]*answer\s*:|rationale\s*:|explanation\s*:)',re.I)
SEEKING=re.compile(r'\b(?:gold|correct|expected|reference|stated)[\s/()-]*(?:gold|correct|standard)?[\s/()-]*answers?\b|\bground[ -]*truth\b|\banswer[ -]*key\b|\bbenchmark[ -]*(?:answers?|questions?|dataset)\b|\bbrowsecomp\b',re.I)
# Hosted datasets and evaluation logs publish run transcripts whose snippets can
# stay under the question-share threshold, so these are matched by URL and title.
EVAL_SOURCE=re.compile(r'(?:huggingface\.co|kaggle\.com)/datasets/|·\s*Datasets at\b|\beval(?:uation)?[-_ ]logs?\b|\bsimple-evals\b|\bbrowsecomp\b',re.I)
# The audit's A list, matched on a view's domain, URL and title. It names kinds of
# host, so an ordinary page on one of them is withheld as well.
A_MARKERS=('huggingface.co','kaggle.com','observation-masking-eval-logs','eval-logs','eval_logs','benchmark-bcplus',
    'browsecomp','browse_comp','fineverify','dr-dci','trajectories','answer key','answer_key','simple-evals',
    'benchmark solution','benchmark_solution','solution trace','solution_trace','evaluation logs','evaluation_logs',
    'evaluation-logs','scored run')
# The audit's C labels, which also accept underscores and a full-width colon.
AUDIT_LABEL=re.compile(r'\bcandidate[\s_-]+answer\s*[:：]|\bgold[\s_-]+answer\b|\bground[\s_-]+truth\b|\bcorrect[\s_-]+answer\b|\breference[\s_-]+answer\b|\brationale\s*[:：]|\bexplanation\s*[:：]',re.I)
URLISH=re.compile(r'(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:[/?#][^\s\'"<>)]*)?',re.I)
ENTRY_TITLE=re.compile(r'^\[\d+\] \*\*(.+?)\*\*',re.M)
ENTRY_URL=re.compile(r'^\s*URL: (\S+)',re.M)
FETCH_WITHHELD=re.compile(r'^\[\d+\] URL: (\S+)\n\s+Info: '+re.escape(MARKER),re.M)
SEARCH_TOOLS=frozenset({'web_search','scholar_search'})
# recover_result returns the stored original of a cut result, before any screening.
RECOVERED_TOOLS=frozenset({'recover_result'})

def words(text):
    text=unicodedata.normalize('NFKC',html.unescape(re.sub(r'<[^>]*>',' ',str(text)))).casefold()
    return re.findall(r'\w+',text)

def raw_words(text):
    # The audit's own tokens, without the markup and entity clean-up above.
    return re.findall(r'\w+',str(text).casefold())

def host(url):
    try:return (urlsplit(str(url)).hostname or '').casefold()
    except ValueError:return ''

def names_evaluation_source(url='',title=''):
    metadata=unquote(host(url)+' '+str(url)+' '+str(title)).casefold()
    return any(marker in metadata for marker in A_MARKERS) or bool(EVAL_SOURCE.search(metadata))

def labelled(text):
    return bool(LABEL.search(text) or AUDIT_LABEL.search(text))

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
        # Both tokenizations, so a run the audit counts is never split here by clean-up.
        self.windows=[]
        for tokenize in (words,raw_words):
            tokens=tokenize(question)
            if not tokens:continue
            width=-(-len(tokens)*QUESTION_SHARE[0]//QUESTION_SHARE[1])
            self.windows.append((tokenize,width,{tuple(tokens[i:i+width]) for i in range(len(tokens)-width+1)}))
        self.width=self.windows[0][1] if self.windows else 0
        self.qhash=hashlib.sha256(question.encode()).hexdigest()
        self.output=Path(output);self.lock=threading.Lock()
        self.quarantine=set();self.recorded=set()

    def is_answer_source(self,text):
        for tokenize,width,spans in self.windows:
            tokens=tokenize(text)
            if any(tuple(tokens[i:i+width]) in spans for i in range(len(tokens)-width+1)):return True
        return False

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

    def refuses_url(self,url):
        return url_key(url) in self.quarantine or names_evaluation_source(url)

    def refuses_command(self,command):
        # A shell command or a download names its source only in its own text.
        text=str(command)
        return self.seeks_answer(text) or any(self.refuses_url(token) for token in URLISH.findall(text))

    def withhold_request(self,name,value):
        text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,default=str)
        self.record(text,'answer_seeking_request:'+str(name))
        return SEEK_MARKER

    def filter_entry(self,entry):
        keys=entry_keys(entry)
        quarantined=bool(keys&self.quarantine)
        title,url=ENTRY_TITLE.search(entry),ENTRY_URL.search(entry)
        evaluation=names_evaluation_source(url.group(1) if url else '',title.group(1) if title else '')
        # The audit reads title and snippet as one text, without the URL line between them.
        copied=self.is_answer_source(entry) or self.is_answer_source(ENTRY_URL.sub(' ',entry))
        if not (quarantined or evaluation or copied or labelled(entry)):return entry
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
        if any(self.refuses_url(url) for url in urls):
            self.record(text,'fetch_quarantined_source')
            return MARKER
        filtered=self.filter(text,stage)
        if filtered==MARKER:self.quarantine|={url_key(url) for url in urls}
        self.quarantine|={url_key(url) for url in FETCH_WITHHELD.findall(filtered)}
        return filtered

    def filter(self,text,stage):
        if not isinstance(text,str) or not (self.is_answer_source(text) or labelled(text)):return text
        self.record(text,stage)
        return MARKER

    def screen(self,name,text,args=None):
        if args and self.seeks_answer(args):return self.withhold_request(name,args)
        if name in SEARCH_TOOLS:return self.filter_search(text)
        if name in RECOVERED_TOOLS:text=self.filter_search(text)
        return self.filter_fetch(text,args,'tool_result:'+name)

def install_guard(question,output):
    from workflows.agent_team import convergence, subagent_runtime
    web_fetch_aligned=importlib.import_module('plugins.tools.web_fetch_aligned')
    web_fetch=importlib.import_module('plugins.tools.web_fetch')
    bash_tool=importlib.import_module('plugins.tools.bash')
    download_tool=importlib.import_module('plugins.tools.download_file')
    from frontier_agent.infra import summary_llm
    guard=AnswerLabelGuard(question,output)
    original_process=subagent_runtime.SwarmToolResultPostProcessor.process
    original_observe=convergence.ConvergenceState.observe_main_tool
    original_aligned=web_fetch_aligned._extract_info_with_llm
    original_summary=summary_llm.summarize
    original_fetch_alias=web_fetch._summary_llm_summarize
    original_single=web_fetch_aligned._fetch_single
    original_one=web_fetch._fetch_one
    original_write_error=bash_tool.bash_output_write_error
    original_download_run=download_tool.arun_sandbox_cmd

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

    # The question reaches the summarizer only with answer leads on. A withheld or filtered page
    # never reaches it, so a lead can only come from content this guard let through.
    async def aligned(content,info_to_extract,truncate_last_num_chars=-1,question=''):
        if guard.seeks_answer(info_to_extract):
            return {'success':True,'extracted_info':guard.withhold_request('web_fetch',info_to_extract),'error':''}
        filtered=guard.filter(content,'aligned_fetch_before_summary')
        if filtered!=content:return {'success':True,'extracted_info':filtered,'error':''}
        return await original_aligned(content,info_to_extract,truncate_last_num_chars,question=question)

    async def summary(content,focus):
        filtered=guard.filter(content,'web_fetch_before_summary')
        if filtered!=content:return filtered
        return await original_summary(content,focus)

    # Screening the delivered result is too late for an evaluation page: by then the
    # reader has fetched it and the page summarizer, also a model, has read it.
    def refused(url):
        if not guard.refuses_url(url):return False
        guard.record(str(url),'fetch_refused_before_read');return True

    async def single(url,info_to_extract,custom_headers=None):
        if refused(url):return MARKER
        return await original_single(url,info_to_extract,custom_headers)

    async def one(url,info_to_extract):
        if refused(url):return MARKER
        return await original_one(url,info_to_extract)

    # Shell commands and downloads reach the network outside the page-read tools,
    # so they are refused before they run. Their output is still screened afterwards.
    def command_refused(command,stage):
        if not guard.refuses_command(command):return False
        guard.record(str(command),stage);return True

    def write_error(command):
        if command_refused(command,'bash_refused_before_run'):return MARKER
        return original_write_error(command)

    async def download_run(sandbox,command,*args,**kwargs):
        # download_file reports a raised error to the model as a failed download.
        if command_refused(command,'download_refused_before_run'):raise RuntimeError(MARKER)
        return await original_download_run(sandbox,command,*args,**kwargs)

    subagent_runtime.SwarmToolResultPostProcessor.process=process
    convergence.ConvergenceState.observe_main_tool=observe
    web_fetch_aligned._extract_info_with_llm=aligned
    web_fetch_aligned._fetch_single=single
    web_fetch._fetch_one=one
    bash_tool.bash_output_write_error=write_error
    download_tool.arun_sandbox_cmd=download_run
    summary_llm.summarize=summary
    web_fetch._summary_llm_summarize=summary
    def restore():
        subagent_runtime.SwarmToolResultPostProcessor.process=original_process
        convergence.ConvergenceState.observe_main_tool=original_observe
        web_fetch_aligned._extract_info_with_llm=original_aligned
        web_fetch_aligned._fetch_single=original_single
        web_fetch._fetch_one=original_one
        bash_tool.bash_output_write_error=original_write_error
        download_tool.arun_sandbox_cmd=original_download_run
        summary_llm.summarize=original_summary
        web_fetch._summary_llm_summarize=original_fetch_alias
    return guard,restore
