Build the docker image: `docker build -t elicutler/nysparcs .`  

Run the docker image: `docker run --rm "$(docker image list elicutler/nysparcs -q)" <command to execute within nysparcs conda env>`

For API access to data, create `.config/secrets.json`, with the following structure:
```
{
  "socrata": {
    "app_token": "<your socrata app token>"
  }
}
