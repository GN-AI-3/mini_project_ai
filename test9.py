import time
from diffusers import StableDiffusionPipeline
import torch

# Record the start time
start_time = time.time()

# Initialize the model pipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generate the image
prompt = "A cartoon of a businesswoman in a modern office."
image = pipe(prompt).images[0]  

# Save the image
image.save("astronaut_rides_horse.png")

# Calculate the time taken
end_time = time.time()
elapsed_time = end_time - start_time

# Print the time taken
print(f"Time taken to generate and save the image: {elapsed_time:.2f} seconds")
