import openai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def call_gpt4o_with_image(image_path, query):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }}
            ]}
        ]
    )

    return response['choices'][0]['message']['content']

result = call_gpt4o_with_image("FPT_3year_prediction.png", """
Phân tích biểu đồ dự đoán giá cổ phiếu FPT trong ảnh. Biểu đồ gồm:
1. Dữ liệu giá cổ phiếu lịch sử (màu xanh dương).
2. Đường dự đoán giá cổ phiếu trong 3 năm tiếp theo (màu đỏ).
3. Thời điểm bắt đầu dự đoán (đường kẻ xanh đứt nét).

Yêu cầu:
1. Tóm tắt xu hướng giá cổ phiếu FPT trong quá khứ (trong biểu đồ).
2. Nhận xét về điểm bắt đầu dự đoán và lý do tại sao có sự sụt giảm ngay trước đó.
3. Đánh giá xu hướng dự đoán: tăng/giảm? tốc độ nhanh/chậm? có dấu hiệu chững lại không?
4. Dự đoán mức giá tiệm cận trong 3 năm tới và ý nghĩa đầu tư.
5. Gợi ý người dùng trong việc có thực sự nên mua không
                               
Trả lời ngắn gọn, logic, rõ ràng.
""")
print(result)
