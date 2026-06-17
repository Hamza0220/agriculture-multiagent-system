from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from .orchestrator import orchestrate
from .crop_doctor import diagnose
from .irrigation_advisor import advise_irrigation
from .market_price_agent import get_market_advice
from .response_synthesizer import synthesize_response

load_dotenv()


try:
    from rag.retriever import retrieve_crop_knowledge
except ImportError:
    def retrieve_crop_knowledge(query, crop_name=None, category=None, top_k=5):
        return "RAG knowledge base not yet available. Using general crop knowledge."


def _run_crop_doctor(orchestration, image_base64):
    crop_name = orchestration.get("crop_detected")
    location = orchestration.get("location", "Pakistan")
    season = orchestration.get("season")
    context = orchestration.get("context_for_agents", "")

    rag_context = retrieve_crop_knowledge(
        query=context,
        crop_name=crop_name,
        category="DISEASE",
        top_k=5,
    )

    return diagnose(
        farmer_description=context,
        crop_name=crop_name,
        image_base64=image_base64,
        location=location,
        season=season,
        weather_summary="Weather data not fetched for this request",
        rag_context=rag_context,
    )


def _run_irrigation_advisor(orchestration):
    crop_name = orchestration.get("crop_detected")
    crop_urdu = orchestration.get("crop_urdu")
    location = orchestration.get("location", "Pakistan")
    context = orchestration.get("context_for_agents", "")

    rag_context = retrieve_crop_knowledge(
        query=context,
        crop_name=crop_name,
        category="IRRIGATION",
        top_k=5,
    )

    return advise_irrigation(
        crop_name=crop_name,
        crop_urdu=crop_urdu,
        location=location,
        field_size_acres=1,
        irrigation_type="tube well",
        rag_irrigation_context=rag_context,
    )


def _run_market_price_agent(orchestration):
    crop_name = orchestration.get("crop_detected", "crop")
    crop_urdu = orchestration.get("crop_urdu")
    location = orchestration.get("location", "Pakistan")

    return get_market_advice(
        crop_name=crop_name,
        crop_urdu=crop_urdu,
        location=location,
    )


def run_agri_pipeline(
    user_query: str,
    image_base64: str = None,
    location: str = "Pakistan",
    crop_name: str = None,
    conversation_history: list = None,
) -> dict:
    orchestration = orchestrate(
        user_query=user_query,
        has_image=image_base64 is not None,
        location=location,
        conversation_history=conversation_history,
    )

    agents_to_call = orchestration.get("agents_to_call", ["CROP_DOCTOR"])

    crop_doctor_result = None
    irrigation_result = None
    market_price_result = None

    agent_tasks = {}

    if "CROP_DOCTOR" in agents_to_call:
        agent_tasks["CROP_DOCTOR"] = lambda: _run_crop_doctor(orchestration, image_base64)

    if "IRRIGATION_ADVISOR" in agents_to_call:
        agent_tasks["IRRIGATION_ADVISOR"] = lambda: _run_irrigation_advisor(orchestration)

    if "MARKET_PRICE" in agents_to_call:
        agent_tasks["MARKET_PRICE"] = lambda: _run_market_price_agent(orchestration)

    if agent_tasks:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(task_fn): agent_name
                for agent_name, task_fn in agent_tasks.items()
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    if agent_name == "CROP_DOCTOR":
                        crop_doctor_result = result
                    elif agent_name == "IRRIGATION_ADVISOR":
                        irrigation_result = result
                    elif agent_name == "MARKET_PRICE":
                        market_price_result = result
                except Exception as e:
                    print(f"[Pipeline] {agent_name} failed: {e}")

    detected_crop = orchestration.get("crop_detected") or crop_name or "crop"
    detected_language = orchestration.get("farmer_language", "roman_urdu")

    response_text = synthesize_response(
        original_query=user_query,
        location=orchestration.get("location", location),
        crop_name=detected_crop,
        language=detected_language,
        crop_doctor_output=crop_doctor_result,
        irrigation_output=irrigation_result,
        market_price_output=market_price_result,
        agents_called=agents_to_call,
    )

    return {
        "response_text": response_text,
        "agents_called": agents_to_call,
        "urgency": orchestration.get("urgency", "MEDIUM"),
        "crop_detected": detected_crop,
        "crop_urdu": orchestration.get("crop_urdu"),
        "orchestration": orchestration,
        "raw_outputs": {
            "crop_doctor": crop_doctor_result,
            "irrigation_advisor": irrigation_result,
            "market_price": market_price_result,
        },
    }
