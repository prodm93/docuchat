   
from unstructured.partition.pdf import partition_pdf

class PDFExtractor:
    def __init__(self, path, fname):  
        self.path = path
        self.fname = fname
        self.raw_pdf_elements = None
        self.tables = []
        self.texts = []

 # Extract elements from PDF
    def extract_pdf_elements(self, extract_images=False):
        """
        Extract images, tables, and chunk text from a PDF file.
        path: File path, which is used to dump images (.jpg)
        fname: File name
        """
        if extract_images:
            self.raw_pdf_elements = partition_pdf(self.fname,
                            extract_images_in_pdf=False,
                            strategy="hi_res",
                            extract_image_block_types=["Image", "Table"],          # optional
                            extract_image_block_to_payload=False,                  # optional
                            extract_image_block_output_dir=self.path,
                            infer_table_structure=False,
                            chunking_strategy="title",
                            max_characters=4000,
                            new_after_n_chars=3800,
                            combine_text_under_n_chars=2000
            )
        else:
            self.raw_pdf_elements = partition_pdf(self.fname,
                        extract_images_in_pdf=False,
                        infer_table_structure=False,
                        chunking_strategy="title",
                        max_characters=4000,
                        new_after_n_chars=3800,
                        combine_text_under_n_chars=2000
        )
        return self.raw_pdf_elements



    # Categorize elements by type
    def categorize_elements(self):
        """
        Categorize extracted elements from a PDF into tables and texts.
        raw_pdf_elements: List of unstructured.documents.elements
        """
        self.raw_pdf_elements = self.extract_pdf_elements() if self.raw_pdf_elements is None else self.raw_pdf_elements
        for element in self.raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                self.tables.append(str(element))
            #elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            elif "Text" in str(type(element)):
                self.texts.append(str(element))
        return self.texts, self.tables
