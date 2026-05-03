Để giải quyết vấn đề này, bạn cần từ bỏ tư duy "bay theo số lượng cố định" và chuyển sang tư duy "duy trì vòng lặp năng lượng" (Energy-based Cycling).
Cách tốt nhất là sử dụng thuật toán Cửa sổ trượt (Sliding Window) hoặc Cuốn chiếu. Cụ thể như sau:

1. Công thức tính số lượng bay an toàn
   Để luôn có con dự phòng thay thế và không bị trống đội hình, số lượng drone bay tối đa (
   ) tại một thời điểm được tính dựa trên thời gian sạc:
   Công thức:
   Nbay​≤​(Tbay​/(Tsạc​+Tnghỉ​))×Ntổng​
   Ví dụ thực tế: Nếu 1 con bay được 20p, sạc mất 40p. Tỷ lệ là 1 bay : 2 sạc.
   Kết quả: Với 10 con, bạn chỉ nên cho bay tối đa 3-4 con cùng lúc để 6-7 con còn lại luôn có đủ thời gian sạc đầy trước khi con cũ về. 2. Cơ chế điều tiết số lượng (4, 6, hay 8 con)
   Để hệ thống tự biết khi nào ra bao nhiêu con, bạn cần lập trình dựa trên Trạng thái sẵn sàng của Trạm sạc:
   Điều kiện xuất kích: Chỉ cho phép một drone cất cánh khi trong trạm có ít nhất 2 con khác đã sạc đầy >90%.
   Hệ thống tự thích nghi:
   Nếu pin sạc nhanh (hoặc trời lặng gió, ít tốn pin): Trạm sạc báo có nhiều con sẵn sàng
   Hệ thống tự đẩy số lượng đang bay lên 6 hoặc 8.
   Nếu pin sạc chậm (hoặc gió to, nhanh hết pin): Số lượng con sẵn sàng ít đi
   Hệ thống tự động giảm số con đang bay xuống còn 4 để đảm bảo không bị "đứt đoạn" vòng lặp. 3. Chiến thuật "Gối đầu" (Staggered Launch)
   Đừng cho 10 con ra cùng lúc. Hãy chia chúng thành các Batch (Đợt):
   Lúc đầu: Cho 4 con ra trước (Đợt 1).
   Sau 10 phút: Cho thêm 2 con ra (Đợt 2).
   Khi Đợt 1 báo pin yếu (30%): Cho 4 con mới từ trạm bay ra thay thế. Khi 4 con mới này đến vị trí, 4 con Đợt 1 mới được về.
   Kết quả: Trên trời luôn có từ 4 đến 6 con, nhưng là các con khác nhau luân phiên. 4. Quy tắc "Mượn trực tiếp" (Hot Swap trên không)
   Để hệ thống tự vận hành mà không cần bạn can thiệp, hãy thiết lập logic:
   Drone A (đang bay): Gửi tín hiệu "Tôi còn 25% pin".
   Server: Kiểm tra trạm sạc, thấy Drone B đã sạc 100%.
   Lệnh: Ra lệnh Drone B cất cánh đi thay vị trí Drone A.
   Xác nhận: Khi Drone B báo "Đã tới vị trí", Drone A mới bắt đầu hành trình quay về sạc.
   💡 Giải pháp kỹ thuật chốt hạ
   Bạn nên cài đặt một Fleet Manager Logic đơn giản:
   Ưu tiên 1: Luôn giữ ít nhất 2 con "Full pin" tại trạm để ứng cứu khẩn cấp.
   Ưu tiên 2: Số lượng đang bay = Tổng số con đã sạc đầy - 2.
   Cảnh báo: Nếu số con sạc đầy < 2, hệ thống tự động gọi bớt các con đang bay về để bảo toàn vòng lặp.
   Nếu bạn làm theo cách này, hệ thống sẽ tự cân bằng: lúc thì bay 4, lúc bay 8 tùy vào tốc độ hồi pin của trạm sạc mà không bao giờ bị tình trạng "hết sạch con để bay".
   Bạn đang dùng phần mềm gì để điều khiển (Mission Planner, DJI SDK, hay tự viết bằng Python/MAVSDK)? Để mình tư vấn sâu hơn về code logic.
