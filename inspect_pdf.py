from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Attention.pdf")
pages = loader.load()

# Print first 2 pages raw so we can see exact text
print("="*60)
print("PAGE 1 RAW TEXT:")
print("="*60)
print(pages[0].page_content)

print("\n" + "="*60)
print("PAGE 2 RAW TEXT:")
print("="*60)
print(pages[1].page_content)

full_text = "\n\n".join([p.page_content for p in pages])
print(full_text[:300])