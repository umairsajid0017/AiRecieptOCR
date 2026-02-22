import json
import re
import time
from datetime import datetime
import torch
from transformers import VisionEncoderDecoderModel, DonutProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 on GPU to save VRAM on the 4GB Quadro M1200
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

class DonutExtractor:
    def __init__(self):
        self.model_name = "mychen76/invoice-and-receipts_donut_v1"
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is None:
            print(f"Loading {self.model_name} on {DEVICE.upper()} ({DTYPE})...")
            
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                torch_dtype=DTYPE,
            ).to(DEVICE)
            self.model.eval()
            
            print(f"Model {self.model_name} loaded successfully on {DEVICE.upper()}!")

    def _xml_to_dict(self, xml_str):
        """Convert the model's XML-like output tags into a Python dict."""
        result = {}
        # Find all top-level sections like <s_header>...</s_header>, <s_items>...</s_items>
        sections = re.findall(r'<s_(\w+?)>(.*?)</s_\1>', xml_str, re.DOTALL)
        
        for section_name, section_content in sections:
            if section_name == 'items':
                # Split items by <sep/> separator
                item_blocks = re.split(r'<sep\s*/>', section_content)
                items = []
                for block in item_blocks:
                    item = {}
                    fields = re.findall(r'<s_(\w+?)>\s*(.*?)\s*</s_\1>', block, re.DOTALL)
                    for field_name, field_value in fields:
                        item[field_name] = field_value.strip()
                    if item:
                        items.append(item)
                result[section_name] = items
            else:
                # For header, summary, etc. — extract individual fields
                fields = re.findall(r'<s_(\w+?)>\s*(.*?)\s*</s_\1>', section_content, re.DOTALL)
                if fields:
                    sub_dict = {}
                    for field_name, field_value in fields:
                        sub_dict[field_name] = field_value.strip()
                    result[section_name] = sub_dict
                else:
                    result[section_name] = section_content.strip()
        
        return result

    def process(self, image):
        self.load_model()
        
        start_time = time.time()
        start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"[DONUT] Inference START: {start_dt}")
        print(f"[DONUT] Device: {DEVICE.upper()}")
        print(f"{'='*60}")
        
        # Prepare the image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)
        
        # Use the model's own decoder start token
        decoder_input_ids = torch.full(
            (1, 1),
            self.model.config.decoder_start_token_id,
            device=DEVICE,
            dtype=torch.long,
        )
        
        # Generate with explicit settings to prevent hanging
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=512,
                num_beams=1,           # Greedy decoding - faster and less VRAM
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        end_time = time.time()
        end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"[DONUT] Inference END:   {end_dt}")
        print(f"[DONUT] Elapsed time:    {elapsed:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Decode the output — keep special tokens so we can parse the XML structure
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        
        # Remove the BOS/EOS tokens but keep the XML-like <s_...> tags
        decoded = decoded.replace("<s>", "").replace("</s>", "").strip()
        
        # Parse the XML-like tags into a proper dict
        parsed = self._xml_to_dict(decoded)
        
        if parsed:
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        
        # Fallback: strip remaining tags and return raw text
        fallback = re.sub(r'<.*?>', ' ', decoded).strip()
        fallback = re.sub(r'\s+', ' ', fallback)
        return fallback
