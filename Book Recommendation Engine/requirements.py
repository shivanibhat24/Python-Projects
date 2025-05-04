# Installation Requirements

## Required Python packages
```
pip install gradio pandas requests langchain llama-cpp-python
```

## LLaMA 3 Model
You'll need to download the LLaMA 3 model (GGUF format) and set the path in the code or environment variable:
- Download the model from Hugging Face: [llama-3-8b-instruct.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/blob/main/llama-3-8b-instruct.Q4_K_M.gguf)
- Set the path in your environment variables:
  ```
  export LLAMA_MODEL_PATH="./llama-3-8b-instruct.Q4_K_M.gguf"
  ```

## Goodreads API Key
1. Create a Goodreads developer account at https://www.goodreads.com/api
2. Apply for an API key
3. Set the API key in your environment variables:
   ```
   export GOODREADS_API_KEY="your_goodreads_api_key"
   ```

Note: The Goodreads API has been officially deprecated. The code includes fallback methods using a sample database, but for a production application, you might need to use web scraping or alternative book APIs.
