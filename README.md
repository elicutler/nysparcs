After building the docker image, to run the container, pass all commands to be executed from the `nysparcs` conda environment into `docker run`.

For API access to data, create `.config/secrets.json`, with the following structure:
```
{
  "socrata": {
    "app_token": "<your socrata app token>"
  }
}
