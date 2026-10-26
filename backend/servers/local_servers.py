"""Three deterministic MCP servers; launch one with --server search|analysis|document."""
import argparse
from mcp.server.fastmcp import FastMCP
from app.orchestration.tools import _analyze_text, _convert_units, _documents, _summarize_dataset

search=FastMCP("research-search")
analysis=FastMCP("safe-analysis")
document=FastMCP("verified-documents")
files=FastMCP("safe-local-files")
text_tools=FastMCP("text-analysis")
dataset=FastMCP("dataset-analysis")
units=FastMCP("unit-conversion")

@search.tool()
def search_documents(query:str)->dict:
    """Search a small controlled set of local research notes."""
    docs=[{"title":"Graph scheduling","text":"DAG schedulers run independent tasks concurrently."},{"title":"MCP","text":"MCP servers expose tools with schemas."},{"title":"Recovery","text":"Bounded retries and validation contain failures."}]
    terms=set(query.lower().split())
    return {"query":query,"results":sorted(docs,key=lambda d:len(terms & set((d['title']+' '+d['text']).lower().split())),reverse=True)[:2]}

@analysis.tool()
def analyze(operation:str,values:list[float])->dict:
    """Calculate sum, mean, min, max, or division for finite values."""
    import math
    if not values or not all(math.isfinite(v) for v in values): raise ValueError("values must be finite and non-empty")
    if operation=="sum": result=sum(values)
    elif operation=="mean": result=sum(values)/len(values)
    elif operation=="min": result=min(values)
    elif operation=="max": result=max(values)
    elif operation=="divide" and len(values)==2 and values[1]!=0: result=values[0]/values[1]
    else: raise ValueError("unsupported operation or division by zero")
    return {"operation":operation,"value":result}

@document.tool()
def create_report(title:str,findings:list[str],sources:list[str])->dict:
    """Create a report from caller-supplied validated findings."""
    return {"title":title,"report":"# "+title+"\n\n"+"\n".join('- '+x for x in findings),"sources":sources}

@files.tool()
def list_documents()->dict:
    """List supported documents in backend/data/user_documents."""
    return _documents({"action":"list"})

@files.tool()
def read_document(filename:str)->dict:
    """Read a supported document from the dedicated user documents folder."""
    return _documents({"action":"read","filename":filename})

@text_tools.tool()
def analyze_text(text:str)->dict:
    """Return word count, character count, reading-time estimate, and frequent terms."""
    return _analyze_text({"text":text})

@dataset.tool()
def summarize_dataset(values:list[float|None],units:list[str]|None=None)->dict:
    """Summarize finite numbers and null values, and report unit-label consistency."""
    return _summarize_dataset({"values":values,"units":units or []})

@units.tool()
def convert_units(value:float,from_unit:str,to_unit:str)->dict:
    """Convert common length, mass, time, or temperature units."""
    return _convert_units({"value":value,"from_unit":from_unit,"to_unit":to_unit})

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--server",choices=["search","analysis","document","files","text","dataset","units"],required=True); server=parser.parse_args().server
    {"search":search,"analysis":analysis,"document":document,"files":files,"text":text_tools,"dataset":dataset,"units":units}[server].run()
