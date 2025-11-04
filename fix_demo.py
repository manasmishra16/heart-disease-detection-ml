#!/usr/bin/env python3
# Fix demo.py by removing corrupt ending

with open('app/demo.py', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Keep only first 693 lines (before corrupt section)
clean_content = ''.join(lines[:693])

# Add clean footer
footer = '''

    # Footer
    st.markdown("---")
    st.info("📋 Medical Disclaimer: This AI system is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.")

if __name__ == "__main__":
    main()
'''

clean_content += footer

with open('app/demo.py', 'w', encoding='utf-8') as f:
    f.write(clean_content)

print("✅ demo.py cleaned successfully!")
print(f"   Kept {len(lines[:693])} lines")
print(f"   Total lines now: {len(clean_content.split(chr(10)))}")
