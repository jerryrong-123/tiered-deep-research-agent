from __future__ import annotations
import asyncio, hashlib, json, os, re, threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal
HERE=Path(__file__).resolve().parent
# prepare.py may lower this per trial; it refuses anything above the sealed 25.
HARD=Decimal(os.environ.get('TRIAL_HARD_CNY') or '25')
TARGET=Decimal('20')
HIDDEN={'reasoning_content','reasoning','reasoning_details','thinking_content','chain_of_thought','private_reasoning','analysis','api_key','apikey','authorization','proxy-authorization','cookie','set-cookie','password','secret','access_token','refresh_token','client_secret'}
_EVENT_LOCK=threading.Lock()
class TrialStop(BaseException):
    def __init__(self, code): self.code = code; super().__init__(code)

def public(value):
    if isinstance(value, dict):
        return {k:public(v) for k,v in value.items() if k.casefold() not in HIDDEN}
    if isinstance(value, list): return [public(v) for v in value]
    return value

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stamp(): return datetime.now(timezone.utc).isoformat()

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def emit(kind, **values):
    # The worker's async transport and to_thread sandbox creation both emit.
    # Serialize open/write/close too: independent append handles lose records
    # on the Windows Docker bind filesystem even for single logical writes.
    record=json.dumps(public({'utc':stamp(),'kind':kind,**values}),ensure_ascii=False,allow_nan=False)+'\n'
    with _EVENT_LOCK:
        with (HERE/'events.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(record)

def multiplier(utc):
    # This one attempt: official pricing verified during the night window.
    if datetime.fromisoformat('2026-10-29T14:10:39+00:00') <= utc < datetime.fromisoformat('2026-10-30T01:00:00+00:00'):
        return Decimal(1)
    local=utc.astimezone(timezone(timedelta(hours=8)))
    # Explicitly checked non-holiday experiment dates. Other dates use peak upper rate.
    if local.date().isoformat() not in {'2026-10-27','2026-10-28'}: return Decimal(2)
    return Decimal(2) if (9<=local.hour<12 or 14<=local.hour<18) else Decimal(1)

def numeric_usage(u):
    if not isinstance(u,dict): return None
    keys=('prompt_tokens','completion_tokens','prompt_cache_hit_tokens')
    if any(type(u.get(k)) is not int or u[k]<0 for k in keys): return None
    if u['prompt_cache_hit_tokens']>u['prompt_tokens']: return None
    return {k:u[k] for k in keys}

# Official offpeak CNY / 1M tokens (cached, uncached, output), checked 2026-10-30.
RATES={'deepseek-flash':(Decimal('.02'),Decimal(1),Decimal(4)),
       'deepseek-v4-pro':(Decimal('.15'),Decimal('4.5'),Decimal('13.5'))}
# Flash reserves the full context. At Pro prices that would hold over CNY 10 per
# call, so Pro reserves the profile's context window, which its guard enforces.
INPUT_UPPER={'deepseek-flash':1048576,'deepseek-v4-pro':262144}

def cost(u, factor, model='deepseek-flash'):
    cached,uncached,output=RATES[model]
    return factor*(Decimal(u['prompt_tokens']-u['prompt_cache_hit_tokens'])*uncached+
        Decimal(u['prompt_cache_hit_tokens'])*cached+
        Decimal(u['completion_tokens'])*output)/Decimal(1000000)

def reservation_amount(model, max_tokens):
    """Peak upper bound: peak prices are twice the offpeak rates above."""
    _,uncached,output=RATES[model]
    return (Decimal(INPUT_UPPER[model])*uncached+Decimal(max_tokens)*output)*2/Decimal(1000000)

# A call cut before its usage arrives holds this, not its full-window reservation. Byte-level BPE gives
# every token at least one byte, so the body actually sent bounds the prompt tokens; the margin covers
# the chat template. Admission still reserves the full window, so a reported charge never exceeds it.
# On 1,401 stored requests whose saved form equals what was sent, the bound was at least 1.73 times the
# reported prompt tokens.
TEMPLATE_MARGIN=4096
PER_MESSAGE=64

def held_upper(payload):
    """Peak upper bound for one cut call, from the request body it sent."""
    model=payload.get('model','deepseek-flash')
    _,uncached,output=RATES[model]
    size=len(json.dumps(payload).encode())+TEMPLATE_MARGIN+PER_MESSAGE*len(payload.get('messages') or [])
    return (Decimal(min(INPUT_UPPER[model],size))*uncached+Decimal(payload['max_tokens'])*output)*2/Decimal(1000000)

class Meter:
    def __init__(self):
        self.spent=Decimal(0); self.held=Decimal(0); self.reserved=Decimal(0)
        self.calls=0; self.lock=asyncio.Lock(); self.entries={}
    def snapshot(self):
        return {'spent_estimate_cny':str(self.spent),'unknown_held_upper_cny':str(self.held),
            'inflight_reserved_upper_cny':str(self.reserved),'target_cny':'20','hard_cap_cny':str(HARD),
            'calls':self.calls,'actual_invoice':None,'price_source':'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
            'rates_cny_per_1M_offpeak':{'cached':'.02','uncached':'1','output':'4'},
            'rates_by_model_cny_per_1M_offpeak':{k:[str(x) for x in v] for k,v in RATES.items()},
            'verified_offpeak_window_utc':{'start':'2026-10-29T14:10:39+00:00','end_exclusive':'2026-10-30T01:00:00+00:00'},'settlement_policy':'maximum start/end timeband; verified night window factor1; otherwise original conservative policy; estimate not invoice',
            'cut_call_held_policy':'request body bound, at most the reservation'}
    def write(self): save(HERE/'MODEL_LEDGER.json',self.snapshot())
    async def reserve(self,payload,role):
        # Reserve the complete supported context at peak uncached price, not a guessed tokenizer count.
        # Settled usage releases the unused reservation. Input+output reservation is conservative.
        model=payload.get('model','deepseek-flash')
        input_upper=INPUT_UPPER[model]
        amount=reservation_amount(model,payload['max_tokens'])
        async with self.lock:
            if self.spent+self.held+self.reserved+amount>HARD: raise TrialStop('model_budget_admission_exhausted')
            ident=self.calls;self.calls+=1;self.reserved+=amount
            self.entries[ident]={'amount':amount,'started':datetime.now(timezone.utc),'input_upper':input_upper,'model':model,
                'held_upper':min(amount,held_upper(payload))}
            self.write();emit('model_admitted',call=ident,role=role,reserved_upper_cny=str(amount))
        return ident
    async def settle(self,ident,usage):
        async with self.lock:
            e=self.entries.pop(ident);self.reserved-=e['amount']
            u=numeric_usage(usage)
            if u is None:
                self.held+=e.get('held_upper',e['amount']); charge=None
            else:
                factor=max(multiplier(e['started']),multiplier(datetime.now(timezone.utc)))
                charge=cost(u,factor,e.get('model','deepseek-flash'))
                if charge>e['amount']:
                    self.spent+=charge;self.write();raise TrialStop('reservation_bound_violated')
                self.spent+=charge
            self.write();emit('model_accounted',call=ident,usage=u,estimated_charge_cny=None if charge is None else str(charge))

def safe_error(exc):
    chain=[]; cursor=exc; seen=set()
    while cursor is not None and id(cursor) not in seen and len(chain)<8:
        seen.add(id(cursor)); item={'type':type(cursor).__module__+'.'+type(cursor).__name__}
        for name in ('errno','winerror'):
            val=getattr(cursor,name,None)
            if type(val) is int:item[name]=val
        chain.append(item);cursor=cursor.__cause__ or cursor.__context__
    message=str(exc).casefold()
    body=getattr(exc,'body',None)
    if isinstance(body,dict) and isinstance(body.get('error'),dict):body=body['error']
    observed_message=message+' '+(str(body.get('message','')).casefold() if isinstance(body,dict) else '')
    reasons=[p for p in ('incomplete chunked read','incomplete message body','peer closed connection',
        'illegal chunk header','read timeout','connection reset','connection refused','insufficient balance',
        'reasoning_content','tool_call_id','must be followed','invalid model','maximum context length') if p in observed_message]
    fields={}
    if isinstance(body,dict):
        for key in ('type','code','param'):
            val=body.get(key)
            if isinstance(val,str) and len(val)<=120 and re.fullmatch(r'[\w.\[\] :-]+',val):fields[key]=val
    status=getattr(exc,'status_code',None)
    return {'exception_chain':chain,'http_status':status if type(status) is int else None,
            'request_id':getattr(exc,'request_id',None),'safe_reason_patterns':reasons,'provider_error_fields':fields}

def balance():
    import urllib.request
    try:
        request=urllib.request.Request('https://api.deepseek.com/user/balance',
            headers={'Authorization':'Bearer '+os.environ['DEEPSEEK_API_KEY']})
        with urllib.request.urlopen(request,timeout=30) as response:data=json.load(response)
        return {'utc':stamp(),'is_available':data.get('is_available'),
            'balances':[{k:r[k] for k in ('currency','total_balance','granted_balance','topped_up_balance') if k in r}
                for r in data.get('balance_infos',[])]}
    except Exception as exc:return {'utc':stamp(),'error':safe_error(exc)}
