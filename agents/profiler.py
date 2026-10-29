"""
agents/profiler.py
------------------
Agent 1  L'Inspecteur.

Reçoit le DataFrame bruité, appelle les outils de profiling
un par un, et retourne un rapport JSON structuré.

C'est le seul agent qui touche directement aux données brutes.
Sa sortie (le rapport JSON) est la seule entrée du Planner.
"""

import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from config import GROQ_API_KEY, MODEL_NAME
from prompts.system_prompts import PROFILER_PROMPT
from tools.profiling_tools import (
    check_missing,
    check_types,
    check_outliers,
    check_duplicates,
    check_cardinality,
)


#  Déclaration des outils pour LangChain 
# LangChain a besoin de fonctions décorées avec @tool pour les passer au LLM.
# On enveloppe nos fonctions pandas dans des fonctions qui acceptent
# un chemin de fichier — le LLM ne peut pas passer un DataFrame directement.

_df_cache: dict[str, pd.DataFrame] = {}  # stocke le DataFrame en mémoire


def _get_df(path: str) -> pd.DataFrame:
    if path not in _df_cache:
        _df_cache[path] = pd.read_csv(path, encoding="utf-8-sig")
    return _df_cache[path]


@tool
def tool_check_missing(csv_path: str) -> str:
    """Check for missing values in the dataset. Returns a JSON report."""
    result = check_missing(_get_df(csv_path))
    return json.dumps(result)


@tool
def tool_check_types(csv_path: str) -> str:
    """Check for type corruption (non-numeric strings in numeric columns)."""
    result = check_types(_get_df(csv_path))
    return json.dumps(result)


@tool
def tool_check_outliers(csv_path: str) -> str:
    """Check for outliers using the IQR method."""
    result = check_outliers(_get_df(csv_path))
    return json.dumps(result)


@tool
def tool_check_duplicates(csv_path: str) -> str:
    """Check for duplicate rows based on key columns."""
    result = check_duplicates(
        _get_df(csv_path),
        key_cols=["date", "gare_depart", "gare_arrivee", "service"]
    )
    return json.dumps(result)


@tool
def tool_check_cardinality(csv_path: str) -> str:
    """Check for high-cardinality columns that may be IDs or free text."""
    result = check_cardinality(_get_df(csv_path))
    return json.dumps(result)


TOOLS = [
    tool_check_missing,
    tool_check_types,
    tool_check_outliers,
    tool_check_duplicates,
    tool_check_cardinality,
]

TOOL_MAP = {t.name: t for t in TOOLS}


#  La boucle ReAct 

def run_profiler(csv_path: str) -> dict:
    """
    Lance l'Inspecteur sur le dataset au chemin csv_path.
    Retourne le rapport JSON parsé en dictionnaire Python.
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
    ).bind_tools(TOOLS)

    # Historique de messages — commence avec la fiche de poste + la mission
    messages = [
        SystemMessage(content=PROFILER_PROMPT),
        HumanMessage(content=f"Profile this dataset: {csv_path}"),
    ]

    print("[Profiler] Starting...")

    # Boucle ReAct : tourne jusqu'à ce que le LLM arrête d'appeler des outils
    while True:
        response = llm.invoke(messages)
        messages.append(response)

        # Si le LLM n'appelle plus d'outil → il a fini, sa réponse est le rapport
        if not response.tool_calls:
            print("[Profiler] Done.")
            break

        # Sinon : exécuter chaque outil demandé et ajouter le résultat
        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            print(f"[Profiler] Calling tool: {tool_name}")

            tool_fn = TOOL_MAP[tool_name]
            result = tool_fn.invoke(tool_args)

            messages.append(
                ToolMessage(content=result, tool_call_id=call["id"])
            )

    # La dernière réponse du LLM est le rapport JSON final
    raw = response.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Si le LLM a ajouté du texte autour du JSON, on l'extrait
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])