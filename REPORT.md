# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> File này điền từ `reports/REPORT_TEMPLATE.md`. Không ghi họ tên, MSSV hay dữ liệu cá nhân.
> Họ tên/MSSV chỉ nằm trong tên repository khi nộp.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1 trên `classification_top5.png` (`sample_id=traffic`): `class_name`=`cab` ; `rank`=1 ; `score`≈0.51 (cột dài nhất, vừa vượt mốc 0.5) ; `taxonomy_name`=`ImageNet-1K` ; `class_id`=468 (mã ImageNet-1K của lớp `cab` trong checkpoint `yolo11n-cls.pt`). Top-5 trên cùng biểu đồ: `cab` > `minibus` > `police_van` > `recreational_vehicle` > `streetcar`.
2. Record này nói về **cả bức ảnh**, không phải một xe riêng. Checkpoint phân loại gán **một** lớp cho toàn ảnh; model không xuất box/mask theo từng vật. Ảnh `traffic` có nhiều xe buýt và ô tô, nhưng top-1 vẫn chỉ là một tên lớp ImageNet.
3. Danh sách tên lớp **đi kèm taxonomy/checkpoint**, không do model tự nghĩ ra. `yolo11n-cls.pt` mang sẵn ImageNet-1K (1000 lớp). Đó là lý do ảnh đường phố COCO lại bị xếp vào `cab` / `minibus` / `streetcar` — các tên đó có trong ImageNet, không phải do model đặt tên mới.
4. Cần cả ba trường vì: `class_id` ổn định khi đổi ngôn ngữ/cách viết; `class_name` để người đọc hiểu; `taxonomy_name` cho biết “cab” ở đây là cab của ImageNet-1K, không phải một lớp nội bộ khác trùng tên.
5. Guideline single-label cần nói rõ **đơn vị gán nhãn là cả ảnh**, tiêu chí chọn một lớp khi có nhiều chủ đề (ví dụ: đối tượng chiếm diện tích lớn nhất / chủ đề chính của cảnh / lớp ưu tiên trong danh sách), và cách xử lý khi không lớp nào phù hợp (đưa ra lớp `other` hoặc escalation — không tự bịa tên ngoài taxonomy).
6. `score` chỉ là độ tin của model khi xếp hạng prediction. Nó không do người gán theo guideline, có thể cao dù lớp sai (ở đây cảnh giao thông nhiều xe buýt nhưng top-1 là `cab`). Ground truth phải do annotator tạo theo quy tắc, rồi reviewer kiểm.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
(Output JSON in ra đầu ô là các dòng `traffic`; mục này dựa trên PNG `kitchen` và dòng so ngưỡng của đúng sample đó.)

1. Record chọn trên `detection_predictions.png`, `sample_id=kitchen`: `class_name`=`person` ; `score`=0.91 ; hộp bao người đứng quay lưng, mặc tạp dề trắng, cạnh lò bên phải. `bbox_format`=`xyxy`, đơn vị pixel, gốc (0, 0) ở góc trên-trái ảnh 640×427. Các prediction khác cùng ngưỡng 0.35 trên PNG: `oven` 0.69 và 0.63, nhiều `bowl` (0.72 / 0.70 / 0.50 / 0.46 / 0.38), `cup` 0.45, và một `person` 0.61 bị cắt mép trái.
2. Hộp `person` 0.91: góc trên-trái gần đỉnh đầu/tóc người đứng; góc dưới-phải gần gót giày. Cạnh trái hộp sát khăn treo / mép tạp dề; cạnh phải sát thân lò `oven` 0.63. Box là hình chữ nhật, nên gồm cả khoảng trống hai bên thân (không bám theo đường cong vai/tạp dề).
3. Trên sample `kitchen` (output ô so sánh): **threshold=0.20 → 17** prediction (`person`, `bowl`, `bowl`, `oven`, `oven`, `person`, `bowl`, `bowl`, `cup`, `cup`, `bowl`, `spoon`, `potted plant`, `spoon`, `dining table`, `spoon`, `bottle`); **threshold=0.35 → 11** prediction (`person`, `bowl`, `bowl`, `oven`, `oven`, `person`, `bowl`, `bowl`, `cup`, `cup`, `bowl`); **threshold=0.60 → 6** prediction (`person`, `bowl`, `bowl`, `oven`, `oven`, `person`). Hai ngưỡng 0.20 và 0.60 chênh 11 hộp.
4. Khi hạ ngưỡng, reviewer phải xem **nhiều hộp hơn**: loại trùng (cùng một vật hai hộp), loại nền/vật nhỏ nhiễu (`spoon`, `potted plant`, `bottle` xuất hiện ở 0.20 nhưng mất ở 0.35), kiểm tra class sai, và không được coi việc model lọc mất object ở ngưỡng cao là “object đó không tồn tại” khi làm ground truth.
5. **Quy tắc box:** Vẽ một hộp `xyxy` nhỏ nhất, song song với cạnh ảnh, bao **toàn bộ phần nhìn thấy** của đúng một instance; nếu vật bị cắt bởi mép ảnh thì cạnh hộp đi sát mép ảnh; không gộp hai vật cùng lớp vào một hộp; không mở rộng hộp để lấy vật che phía trước.
6. Trên `kitchen`, `person` 0.61 ở mép trái: chỉ thấy tay/một phần thân, bị cắt khung hình. Đây là kiểu **cắt mép thường gặp** → viết thành quy tắc guideline (“vẫn gán nếu nhìn thấy đủ để nhận ra người; hộp đến mép ảnh”). Không cần escalation. Ngược lại, chùm nồi/chảo treo trên tường không nằm trong COCO-80 một cách rõ ràng thì mới hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
(Output JSON in ra đầu ô là `traffic-001` / `traffic-002`; mục này dựa trên PNG `kitchen`.)

1. Record chọn trên `segmentation_prediction.png`, `sample_id=kitchen`: `class_name`=`person` ; `score`=0.90 ; mask màu phủ theo thân người đứng (tóc, vai, tạp dề, quần, giày), không chỉ là hình chữ nhật. `instance_id` trong JSON lab có dạng `kitchen-NNN` (ví dụ `kitchen-001` nếu instance này được đánh số đầu). Các instance khác cùng ảnh: `person` 0.43 (mép trái), `dining table` 0.58, `bowl` 0.74 / 0.70 / 0.55 / 0.37, `oven` 0.49, `potted plant` 0.63, `spoon` 0.49.
2. Box chỉ cho biết phạm vi chữ nhật. Polygon/mask thêm **đường biên thật**: chỗ lõm giữa hai chân, mép tạp dề, đường vai — pixel nào thuộc instance, pixel nào là nền hoặc vật khác. Hai `bowl` trên bàn vẫn tách được dù cùng lớp.
3. `instance_id` chỉ để **phân biệt từng object trong bộ output bài lab** (`kitchen-001`, `kitchen-002`, …). Nó không phải `class_id` (vì hai người cùng `class_id=0` / `person` vẫn cần hai ID). Nó cũng không phải tracking ID: lab không theo dõi vật qua nhiều khung hình; ID chỉ có nghĩa trong file JSON lần chạy này.
4. **Quy tắc vẽ biên mask:** Vẽ sát biên **phần nhìn thấy** của đúng một instance; không tô vùng bị che khuất nếu guideline không cho phép suy diễn; hai vật chạm nhau thì đường biên tách ở chỗ tiếp xúc, không gộp mask; lỗ/điểm lõm (khe giữa chân, tay chống hông) để ngoài mask nếu đó là nền.
5. Chỗ khó: mặt bàn (`dining table` 0.58) chồng lên nhiều `bowl`, và `person` 0.43 bị cắt mép trái / một phần bị bàn che. Với cắt mép và hai vật chạm nhau trên bàn → **guideline** (tô phần nhìn thấy, tách instance). Với chậu cây `potted plant` 0.63 lẫn nồi treo góc trên-trái, biên rất mờ → **escalation** cho Lab Coach, không tự quyết lớp/biên cho đống đồ treo.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp (hoặc tập lớp, nếu multi-label) cho **cả ảnh**, lấy từ taxonomy đã chốt | Ảnh `traffic` là cảnh nhiều xe, nhưng top-1 ImageNet là `cab` (~0.51), không phải “cả đường phố” | Đọc guideline + taxonomy, chọn đúng một lớp cho toàn ảnh, không copy top-1 của model | Đúng taxonomy chưa; có chọn nhầm chủ đề phụ không; không lấy `score` làm “điểm nhãn” |
| Phát hiện vật thể | Mỗi instance: một `class_id` COCO-80 + một hộp `xyxy` pixel | Ngưỡng 0.20 thêm `spoon` / `potted plant` / `bottle`; `person` 0.61 cắt mép trái; nhiều `bowl` sát nhau | Vẽ đủ object theo guideline, kể cả vật model bỏ ở ngưỡng 0.60; một hộp / một instance | Thiếu/thừa hộp, hộp lệch, hai vật một hộp, class sai; so với guideline chứ không so với `score` |
| Instance segmentation | Mỗi instance: class + box + polygon/mask riêng | Mask `dining table` phủ cả bàn đang có bát; `potted plant` 0.63 lẫn nồi treo; hai `person` cùng lớp khác instance | Vẽ biên phần nhìn thấy, tách instance cùng lớp (hai `bowl`, hai `person`) | Mask tràn/thiếu, gộp hai object, tô vùng bị che không theo rule; ca mờ thì đã escalation chưa |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai có checksum trong notebook; không tải ảnh cá nhân, khuôn mặt riêng, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub; không ghi họ tên/MSSV/email vào báo cáo hay JSON/PNG.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi: **dừng ngay**, không upload tiếp, và báo **Lab Coach / GV** (kênh hỗ trợ của lớp), không tự thay bằng ảnh khác.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
