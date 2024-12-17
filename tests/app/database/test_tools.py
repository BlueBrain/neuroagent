"""Test of the tool router."""

import pytest

from neuroagent.app.config import Settings
from neuroagent.app.dependencies import get_language_model, get_settings
from neuroagent.app.main import app
from neuroagent.app.routers.database.schemas import ToolCallSchema


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.asyncio
async def test_get_tool_calls(
    patch_required_env, fake_llm_with_tools, httpx_mock, app_client, db_connection
):
    # Put data in the db
    llm, _, _ = fake_llm_with_tools
    app.dependency_overrides[get_language_model] = lambda: llm
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    httpx_mock.add_response(url="https://fake_url/api/nexus/v1/search/query/")

    with app_client as app_client:
        wrong_response = app_client.get("/tools/test/1234")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_output["thread_id"]

        # Fill the thread
        app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
            params={"thread_id": thread_id},
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )

        tool_calls = app_client.get(f"/tools/{thread_id}/wrong_id")
        assert tool_calls.status_code == 404
        assert tool_calls.json() == {"detail": {"detail": "Message not found."}}

        # Get the messages of the thread
        messages = app_client.get(f"/threads/{thread_id}").json()
        message_id = messages[-1]["message_id"]
        tool_calls = app_client.get(f"/tools/{thread_id}/{message_id}").json()

    assert (
        tool_calls[0]
        == ToolCallSchema(
            call_id="call_zHhwfNLSvGGHXMoILdIYtDVI",
            name="get-morpho-tool",
            arguments={
                "brain_region_id": "http://api.brain-map.org/api/v2/data/Structure/549"
            },
        ).model_dump()
    )


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.asyncio
async def test_get_tool_output(
    patch_required_env,
    fake_llm_with_tools,
    app_client,
    httpx_mock,
    db_connection,
):
    # Put data in the db
    llm, _, _ = fake_llm_with_tools
    app.dependency_overrides[get_language_model] = lambda: llm

    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    httpx_mock.add_response(
        url="https://fake_url/api/nexus/v1/search/query/",
        json={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "@id": "https://bbp.epfl.ch/data/bbp/mmb-point-neuron-framework-model/ca1f0e5f-ff08-4476-9b5f-95f3c9d004fd",
                            "brainRegion": {
                                "@id": (
                                    "http://api.brain-map.org/api/v2/data/Structure/629"
                                ),
                                "label": (
                                    "Ventral anterior-lateral complex of the thalamus"
                                ),
                            },
                            "description": (
                                "This is a morphology reconstruction of a mouse"
                                " thalamus cell that was obtained from the Janelia"
                                " Mouselight project"
                                " http://ml-neuronbrowser.janelia.org/ . This"
                                " morphology is positioned in the Mouselight custom"
                                " 'CCFv2.5' reference space, instead of the Allen"
                                " Institute CCFv3 reference space."
                            ),
                            "mType": {"label": "VPL_TC"},
                            "name": "AA0519",
                            "subjectAge": {
                                "label": "60 days Post-natal",
                            },
                            "subjectSpecies": {"label": "Mus musculus"},
                        }
                    }
                ]
            }
        },
    )
    with app_client as app_client:
        wrong_response = app_client.get("/tools/output/test/123")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_output["thread_id"]

        # Fill the thread
        app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
            params={"thread_id": thread_id},
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )

        tool_output = app_client.get(f"/tools/output/{thread_id}/123")
        assert tool_output.status_code == 404
        assert tool_output.json() == {"detail": {"detail": "Tool call not found."}}

        # Get the messages of the thread
        messages = app_client.get(f"/threads/{thread_id}").json()
        message_id = messages[-1]["message_id"]
        tool_calls = app_client.get(f"/tools/{thread_id}/{message_id}").json()
        tool_call_id = tool_calls[0]["call_id"]
        tool_output = app_client.get(f"/tools/output/{thread_id}/{tool_call_id}")

    assert tool_output.json() == [
        {
            "morphology_id": "https://bbp.epfl.ch/data/bbp/mmb-point-neuron-framework-model/ca1f0e5f-ff08-4476-9b5f-95f3c9d004fd",
            "morphology_name": "AA0519",
            "morphology_description": (
                "This is a morphology reconstruction of a mouse thalamus cell that was"
                " obtained from the Janelia Mouselight project"
                " http://ml-neuronbrowser.janelia.org/ . This morphology is positioned"
                " in the Mouselight custom 'CCFv2.5' reference space, instead of the"
                " Allen Institute CCFv3 reference space."
            ),
            "mtype": "VPL_TC",
            "brain_region_id": "http://api.brain-map.org/api/v2/data/Structure/629",
            "brain_region_label": "Ventral anterior-lateral complex of the thalamus",
            "subject_species_label": "Mus musculus",
            "subject_age": "60 days Post-natal",
        }
    ]
