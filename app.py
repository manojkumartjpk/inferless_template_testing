import inferless

from transformers import pipeline
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

@inferless.request
class RequestObjects(BaseModel):
    input_image_url: str = Field(default='https://hello.world')
    count_iterations: int = Field(default=4)
    prompt: str = Field(default="a horse near a beach")
    mask_arr: List[int] = Field(default=[1, 5])
    is_aws: Optional[bool] = None


@inferless.response
class ResponseObjects(BaseModel):
    generated_txt: str = Field(default='Test output')
    count_iterations: int = Field(default=4)
    quality: float = Field(default=0.7)
    positions: List[int] = Field(default=[1, 5])
    is_aws: Optional[float] = Field(default=False)

class InferlessPythonModel:
  
    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)

    def infer(self, inputs):
        prompt = inputs.prompt
        pipeline_output = self.generator(prompt, do_sample=True, min_length=50)
        generated_txt = pipeline_output[0]["generated_text"]
        generateObject = ResponseObjects(generated_txt = generated_txt , count_iterations=1, quality=0.8 , positions=[0,1] )
        return generateObject

    # perform any cleanup activity here
    def finalize(self,args):
        self.pipe = None
