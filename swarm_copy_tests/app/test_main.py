def test_settings_endpoint(app_client, dont_look_at_env_file, settings):
    response = app_client.get("/settings")

    replace_secretstr = settings.model_dump()
    replace_secretstr["keycloak"]["password"] = "**********"
    replace_secretstr["openai"]["token"] = "**********"
    assert response.json() == replace_secretstr


def test_readyz(app_client):
    response = app_client.get(
        "/",
    )

    body = response.json()
    assert isinstance(body, dict)
    assert body["status"] == "ok"
