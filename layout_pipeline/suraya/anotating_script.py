from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import torch
from pdf2image import convert_from_path
import os
import io
import img2pdf
import gc
from PIL import Image

def pdf_to_images(path, dpi=300):
    images = []
    if os.path.isfile(path):
        images.extend(convert_from_path(path, dpi=dpi))
    else:
        for pdf_file in tqdm(os.listdir(path), desc="Converting PDFs"):
            if pdf_file.lower().endswith('.pdf'):
                pdf_path = os.path.join(path, pdf_file)
                images.extend(convert_from_path(pdf_path, dpi=dpi))
    print(f"\n✅ Conversion complete!")
    return images

pages = pdf_to_images("/Users/maryamsaad/Documents/arabic-ocr-transformer/من معتمدات الكتب/المغني لابن قدامة طبعة عالم الكتب 15مجلد/mogni01p.pdf")
device = "mps" if torch.backends.mps.is_available() else "cpu"

foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor(device=device)
CONF_THRESHOLD = 0.65
detection_predictor.to(device)

pdf_list=[ "/Users/maryamsaad/Documents/arabic-ocr-transformer/من معتمدات الكتب/البحر المحيط ـ الزركشي 6 مجلدات/bmoheet3.pdf", 
          "/Users/maryamsaad/Documents/arabic-ocr-transformer/من معتمدات الكتب/المجموع شرح المهذب 23 مجلد/magm08.pdf",
            "/Users/maryamsaad/Documents/arabic-ocr-transformer/من معتمدات الكتب/المغني لابن قدامة طبعة عالم الكتب 15مجلد/mogni07.pdf",
              "من معتمدات الكتب/بداية المجتهد 6 مجلد/بداية المجتهد ونهاية المقتصد 4.pdf",
                "/Users/maryamsaad/Documents/arabic-ocr-transformer/من معتمدات الكتب/سبل الهدى والرشاد في سيرة خير العباد 12 مجلد/suho04.pdf",
                "/Users/maryamsaad/Documents/arabic-ocr-transformer/من معتمدات الكتب/فتح الباري شرح صحيح البخاري – دار السلام، الرياض – 14مجلد/فتح الباري شرح صحيح البخاري – دار السلام، الرياض – جلد 08.pdf",
                "/Users/maryamsaad/Documents/arabic-ocr-transformer/من معتمدات الكتب/لسان العرب 15 مجلد/لسان العرب (4).pdf"]


os.makedirs("annotated_pdfs", exist_ok=True)

def annotate_pdf(book):
    pages = pdf_to_images(book)
    output_images = []

    for page_index, image in enumerate(tqdm(pages, desc=f"Annotating {os.path.basename(book)}")):
        detection_results = detection_predictor([image])
        det_result = detection_results[0]
        sorted_boxes = sorted(det_result.bboxes, key=lambda b: b.bbox[0], reverse=True)
        sorted_boxes = [box for box in sorted_boxes if box.confidence >= CONF_THRESHOLD]


        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(image)
        for i, bbox in enumerate(sorted_boxes):
            x1, y1, x2, y2 = map(int, bbox.bbox)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{i+1}", color="yellow", fontsize=10, backgroundcolor="black")
        ax.axis("off")
        plt.tight_layout()

        # Convert figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
        buf.seek(0)
        annotated_image_path = f"temp_page_{page_index}.jpg"
        with Image.open(buf).convert("RGB") as im:
            im.save(annotated_image_path, "JPEG", quality=85)
        output_images.append(annotated_image_path)

        # Free memory
        plt.close(fig)
        plt.close("all")
        gc.collect()

    # Merge all annotated images into a PDF
    output_pdf = f"annotated_pdfs/{os.path.basename(book).replace('.pdf', '')}_annotated.pdf"
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(output_images))
    print(f"✅ PDF saved successfully as {output_pdf}")

    # Clean up temp images
    for p in output_images:
        os.remove(p)


# Process all PDFs
for book in tqdm(pdf_list, desc="Processing PDFs"):
    try:
        annotate_pdf(book)
    except Exception as e:
        print(f"⚠️ Error processing {book}: {e}")
        continue
