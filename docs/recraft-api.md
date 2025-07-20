# Recraft Vectorize API Details

Vectorize Image
Converts a given raster image to SVG format.

```
POST https://external.api.recraft.ai/v1/images/vectorize
```

Example:
```
response = client.post(
    path='/images/vectorize',
    cast_to=object,
    options={'headers': {'Content-Type': 'multipart/form-data'}},
    files={'file': open('image.png', 'rb')},
)
print(response['image']['url'])
```

Output:
https://img.recraft.ai/fZm6nwEjI9Qy94LukIKbxRm4w2i5crwqu459qKg7ZWY/rs:fit:1341:1341:0/raw:1/plain/abs://external/images/2835e19f-282b-419b-b80c-9231a3d51517

Parameters:
- Body of the request should be a file in PNG format and parameters passed as content type 'multipart/form-data'.
- The image must be less than 5 MB in size, have resolution less than 16 MP, and max dimension less than 4096 pixels.

| Parameter	  |      Type	|                Description |
| --- | --- | --- |
| response_format	|    string or null, default is url | 	The format in which the generated images are returned. Must be one of 'url' or 'b64_json'. |

---