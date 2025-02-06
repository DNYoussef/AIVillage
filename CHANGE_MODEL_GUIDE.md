**Guide to Changing the Compressed Model in AI Village**

This guide provides step-by-step instructions to change the model being compressed within the AI Village system. Follow these steps to ensure a smooth transition to your desired model.

---

### **1. Backup Existing Configuration and Scripts**

Before making any changes, back up the current configuration files and scripts to prevent data loss.

- **Files to Backup:**
  - `run_compression.py`
  - `config/unified_config.py`
  - `config/default.yaml`
  - `config/openrouter_agents.yaml`

- **Backup Commands (Windows Command Prompt):**
  ```cmd
  copy run_compression.py run_compression_backup.py
  copy config\unified_config.py config\unified_config_backup.py
  copy config\default.yaml config\default_backup.yaml
  copy config\openrouter_agents.yaml config\openrouter_agents_backup.yaml
  ```

---

### **2. Identify Current Model Configuration**

The model to be compressed is specified in the `run_compression.py` script and managed through the `UnifiedConfig` system.

- **Key File:**
  - `run_compression.py`

- **Relevant Code Snippet:**
  ```python
  model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
  ```

---

### **3. Update the Model Name in `run_compression.py`**

To change the model being compressed, modify the `model_name` variable in the `run_compression.py` script.

- **Steps:**
  1. Open `run_compression.py` in a text editor.
  2. Locate the line defining `model_name`.
  3. Replace the existing model name with the desired model's identifier or file path.

- **Example:**
  - **Original:**
    ```python
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    ```
  - **Updated:**
    ```python
    model_name = "Your/Desired-Model-Name-Version"
    ```

- **Notes:**
  - Ensure the new model name corresponds to a valid model available in your environment or accessible via the specified path.
  - If the model is hosted on platforms like Hugging Face, verify the identifier's correctness.

---

### **4. Update Configuration Files if Necessary**

Ensure that other configuration files support the new model.

- **Files to Check:**
  - `config/unified_config.py`
  - `config/default.yaml`
  - `config/openrouter_agents.yaml`

- **Steps:**
  1. **`config/unified_config.py`:**
     - Ensure agent configurations reference the new model.
     - Example for updating agents:
       ```yaml
       agents:
         king:
           frontier_model: "Your/Desired-Frontier-Model"
           local_model: "Your/Desired-Local-Model"
           description: "Strategic planning and decision making agent"
           settings:
             temperature: 0.7
             max_tokens: 1000
             system_prompt: |
               You are King, an advanced AI agent specializing in complex problem-solving and strategic thinking...
         # Repeat for other agents (sage, magi) as necessary
       ```

  2. **`config/default.yaml`:**
     - Update any model defaults if required.
     - Verify parameters align with the new model.

  3. **`config/openrouter_agents.yaml`:**
     - Ensure agent-specific settings reference the new model.
     - Example:
       ```yaml
       agents:
         king:
           frontier_model: "Your/Desired-Frontier-Model"
           local_model: "Your/Desired-Local-Model"
           description: "Strategic planning and decision making agent"
           settings:
             temperature: 0.7
             max_tokens: 1000
             system_prompt: |
               You are King, an advanced AI agent specializing in complex problem-solving and strategic thinking...
         # Repeat for other agents (sage, magi) as necessary
       ```

---

### **5. Ensure Model Compatibility**

Verify that the new model is compatible with the compression tools and configurations.

- **Dependencies:**
  - Update `requirements.txt` if the new model requires additional packages.
    ```cmd
    pip install -r requirements.txt
    ```

- **Tokenizer Compatibility:**
  - Confirm that the tokenizer in `run_compression.py` matches the new model.
  - Example:
    ```python
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ```

- **Device Configuration:**
  - Verify the `device` parameter in configurations (e.g., `'cuda'` for GPU).

- **Hardware Requirements:**
  - Ensure your system meets the new model's hardware requirements, especially for larger models.

---

### **6. Update `run_compression.py` if Necessary**

If the new model has different initialization requirements, adjust the script accordingly.

- **Steps:**
  1. **Tokenizer Initialization:**
     - Ensure the tokenizer corresponds to the new model.
     - Example:
       ```python
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       ```

  2. **Model Loading:**
     - Verify the model loading command is appropriate.
     - Example:
       ```python
       model = AutoModelForCausalLM.from_pretrained(model_name)
       ```

  3. **Compression Configuration:**
     - Update `CompressionConfig` if the new model requires different settings.
     - Example:
       ```python
       compression_config = CompressionConfig.from_model(model)
       ```

---

### **7. Execute the Compression Task**

Run the compression script to compress the new model.

- **Steps:**
  1. Open Command Prompt and navigate to the working directory:
     ```cmd
     cd c:\Users\17175\Desktop\AIVillage
     ```

  2. Execute the compression script:
     ```cmd
     python run_compression.py
     ```

  3. Monitor the logs for any errors or warnings to ensure the process runs smoothly.

- **Expected Output:**
  - Logs indicating the stages of compression.
  - Successfully saved compressed model files (e.g., `compressed_model.pt`).
  - Compression metrics displayed in the logs.

---

### **8. Validate the Compressed Model**

Ensure that the compressed model functions as expected.

- **Steps:**
  1. **Load the Compressed Model:**
     - Use the `InferenceEngine` to load and test the compressed model.
     ```python
     inference_engine = InferenceEngine(compressed_state, inference_config)
     ```

  2. **Run Test Generations:**
     - Provide sample prompts to verify model responses.
     ```python
     test_prompt = "Explain the theory of relativity."
     input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.cuda()
     output_ids = inference_engine.generate(input_ids, max_length=100, temperature=0.7)
     response = tokenizer.decode(output_ids)
     print(response)
     ```

  3. **Review Compression Metrics:**
     - Ensure that the compression ratio and other metrics meet your requirements.

---

### **9. Update Documentation and Configuration Management**

Maintain accurate documentation and ensure that configuration management reflects the changes.

- **Steps:**
  1. **Documentation:**
     - Update internal documentation to reflect the change in the compressed model.
     - Example updates in `docs/user_guide.md` or `docs/architecture.md`.

  2. **Version Control:**
     - Commit the changes to version control with a meaningful message.
     ```cmd
     git add run_compression.py config\*
     git commit -m "Change compressed model to [New Model Name]"
     git push
     ```

---

### **10. Troubleshooting**

If issues arise during or after the compression process, consider the following steps:

- **Common Issues:**
  - **Model Loading Errors:**
    - Ensure the model name/path is correct.
    - Verify that all dependencies are installed.
  
  - **Compression Failures:**
    - Check system resources (CPU, GPU, memory).
    - Review logs for specific error messages.
  
  - **Compatibility Issues:**
    - Confirm that the new model is compatible with compression tools and configurations.

- **Steps:**
  1. **Review Logs:**
     - Examine log outputs for error messages or warnings.

  2. **Consult Documentation:**
     - Refer to AI Village documentation for specific configuration parameters and troubleshooting.

  3. **Seek Support:**
     - Reach out to the development team or community forums for assistance if issues persist.

---

By following this guide, you should be able to successfully change the model being compressed within the AI Village system. Ensure that all configurations are consistent across relevant files and that your system resources are adequately provisioned for the new model.