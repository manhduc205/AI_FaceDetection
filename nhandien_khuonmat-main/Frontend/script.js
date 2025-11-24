const BASE_URL = "https://7497-118-70-74-174.ngrok-free.app"; // Thay bằng URL ngrok thực tế

console.log("script.js loaded");

const navUpload = document.getElementById("nav-upload");
const navSchedule = document.getElementById("nav-schedule");
const navAttendanceList = document.getElementById("nav-attendance-list");
const pageUpload = document.getElementById("page-upload");
const pageSchedule = document.getElementById("page-schedule");
const pageAttendance = document.getElementById("page-attendance");
const pageAttendanceList = document.getElementById("page-attendance-list");
const scheduleTable = document.getElementById("scheduleTable");
const studentList = document.getElementById("studentList");
const subjectName = document.getElementById("subjectName");
const className = document.getElementById("className");
const streamFrame = document.getElementById("streamFrame");
const videoContainer = document.getElementById("videoContainer");
const saveAttendanceBtn = document.getElementById("saveAttendance");
const startFaceRecognitionBtn = document.getElementById("startFaceRecognition");
const stopFaceRecognitionBtn = document.getElementById("stopFaceRecognition");
const sidebar = document.querySelector(".sidebar");
const statusModal = new bootstrap.Modal(document.getElementById("statusModal"));
const statusIcon = document.getElementById("statusIcon");
const statusMessage = document.getElementById("statusMessage");
const classSelect = document.getElementById("classSelect");
const subjectSelect = document.getElementById("subjectSelect");
const sessionList = document.getElementById("sessionList");

let recognizedStudents = new Set();
let currentClassId = null;
let currentTimetableId = null;
let studentMap = new Map();
let recognitionInterval = null;
let isSaving = false;

if (!pageUpload) console.error("Không tìm thấy pageUpload!");
if (!statusModal) console.error("Không tìm thấy statusModal!");
if (!statusIcon) console.error("Không tìm thấy statusIcon!");
if (!statusMessage) console.error("Không tìm thấy statusMessage!");
if (!saveAttendanceBtn) console.error("Không tìm thấy saveAttendance!");
if (!startFaceRecognitionBtn)
  console.error("Không tìm thấy startFaceRecognition!");
if (!stopFaceRecognitionBtn)
  console.warn("Không tìm thấy stopFaceRecognitionBtn, bỏ qua!");

function showPage(page) {
  [pageUpload, pageSchedule, pageAttendance, pageAttendanceList].forEach(
    (p) => p && p.classList.add("hidden")
  );
  page && page.classList.remove("hidden");
  if (videoContainer) {
    videoContainer.classList.add("hidden");
  }
  if (streamFrame) {
    streamFrame.src = "";
  }
  if (sidebar) {
    sidebar.classList.remove("sidebar-collapsed");
  }
  if (recognitionInterval) {
    clearInterval(recognitionInterval);
    recognitionInterval = null;
  }
  if (sessionList) {
    sessionList.innerHTML = "";
  }
}

if (navUpload) {
  navUpload.onclick = () => {
    showPage(pageUpload);
    navUpload.classList.add("active");
    navSchedule && navSchedule.classList.remove("active");
    navAttendanceList && navAttendanceList.classList.remove("active");
  };
}

if (navSchedule) {
  navSchedule.onclick = () => {
    showPage(pageSchedule);
    navUpload && navUpload.classList.remove("active");
    navSchedule.classList.add("active");
    navAttendanceList && navAttendanceList.classList.remove("active");
    fetchSchedule();
  };
}

if (navAttendanceList) {
  navAttendanceList.onclick = () => {
    showPage(pageAttendanceList);
    navUpload && navUpload.classList.remove("active");
    navSchedule && navSchedule.classList.remove("active");
    navAttendanceList.classList.add("active");
    fetchClasses();
  };
}

async function fetchClasses() {
  try {
    const res = await fetch(`${BASE_URL}/api/classes`);
    const data = await res.json();
    if (classSelect) {
      classSelect.innerHTML = '<option value="">-- Chọn giá trị lớp --</option>';
      data.forEach((cls) => {
        const option = document.createElement("option");
        option.value = cls.ID;
        option.textContent = cls.NameClass;
        classSelect.appendChild(option);
      });
      subjectSelect.disabled = true;
      subjectSelect.innerHTML = '<option value="">-- Chọn môn --</option>';
    }
  } catch (error) {
    console.error("Error fetching classes:", error);
    if (classSelect) {
      classSelect.innerHTML =
        '<option value="">Lỗi khi tải danh sách lớp</option>';
    }
  }
}

classSelect.addEventListener("change", async () => {
  const classId = classSelect.value;
  if (classId) {
    try {
      const res = await fetch(`${BASE_URL}/api/class/${classId}/subjects`);
      const data = await res.json();
      subjectSelect.disabled = false;
      subjectSelect.innerHTML = '<option value="">-- Chọn môn --</option>';
      data.forEach((subject) => {
        const option = document.createElement("option");
        option.value = subject.ID;
        option.textContent = subject.Name_Subject;
        subjectSelect.appendChild(option);
      });
    } catch (error) {
      console.error("Error fetching subjects:", error);
      subjectSelect.innerHTML =
        '<option value="">Lỗi khi tải danh sách môn</option>';
    }
  } else {
    subjectSelect.disabled = true;
    subjectSelect.innerHTML = '<option value="">-- Chọn môn --</option>';
    sessionList.innerHTML = "";
  }
});

subjectSelect.addEventListener("change", async () => {
  const classId = classSelect.value;
  const subjectId = subjectSelect.value;
  if (classId && subjectId) {
    try {
      const res = await fetch(
        `${BASE_URL}/api/attendance_sessions_by_class_subject/${classId}/${subjectId}`
      );
      const data = await res.json();
      sessionList.innerHTML = "";
      if (data.length === 0) {
        sessionList.innerHTML =
          "<div class='accordion-item'><div class='accordion-body'>Chưa có buổi học nào được điểm danh</div></div>";
        return;
      }
      data.forEach((session, index) => {
        const accordionItem = document.createElement("div");
        accordionItem.className = "accordion-item";
        accordionItem.innerHTML = `
          <h2 class="accordion-header" id="heading${index}">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${index}">
              ${session.date} - ${session.subject}
            </button>
          </h2>
          <div id="collapse${index}" class="accordion-collapse collapse" data-bs-parent="#sessionList">
            <div class="accordion-body">
              <table class="table table-striped table-bordered bg-white mb-0">
                <thead>
                  <tr>
                    <th>Mã SV</th>
                    <th>Họ tên</th>
                    <th>Trạng thái</th>
                  </tr>
                </thead>
                <tbody id="studentList${index}"></tbody>
              </table>
            </div>
          </div>
        `;
        sessionList.appendChild(accordionItem);
        const studentListElement = document.getElementById(
          `studentList${index}`
        );
        session.students.forEach((student) => {
          const tr = document.createElement("tr");
          const statusText =
            student.Status === "Present" ? "Có mặt" : "Vắng mặt";
          tr.innerHTML = `
            <td>${student.MSV}</td>
            <td>${student.FullName}</td>
            <td>${statusText}</td>
          `;
          studentListElement.appendChild(tr);
        });
      });
    } catch (error) {
      console.error("Error fetching attendance sessions:", error);
      sessionList.innerHTML =
        "<div class='accordion-item'><div class='accordion-body'>Lỗi khi tải danh sách buổi học</div></div>";
    }
  } else {
    sessionList.innerHTML = "";
  }
});

async function fetchSchedule() {
  try {
    const res = await fetch(`${BASE_URL}/api/schedule`);
    const data = await res.json();
    console.log("Dữ liệu từ API:", data);
    renderSchedule(data);
  } catch (error) {
    console.error("Error fetching schedule:", error);
    if (scheduleTable) {
      scheduleTable.innerHTML =
        "<tr><td colspan='8'>Lỗi khi tải thời khóa biểu</td></tr>";
    }
  }
}

function renderSchedule(data) {
  if (!Array.isArray(data) || data.length === 0) {
    console.error("Dữ liệu lịch học rỗng");
    if (scheduleTable) {
      scheduleTable.innerHTML =
        '<tr><td colspan="8" class="text-center py-4">Không có dữ liệu lịch học</td></tr>';
    }
    return;
  }

  const normalizeDay = (str) => {
    return str
      .trim()
      .toLowerCase()
      .replace(/(thứ\s*|chủ\s*)/, (match) => match.replace(/\s/g, ""))
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "");
  };

  const daysOfWeek = [
    "Thứ 2",
    "Thứ 3",
    "Thứ 4",
    "Thứ 5",
    "Thứ 6",
    "Thứ 7",
    "Chủ nhật",
  ];

  const timeSlots = Array.from({ length: 11 }, (_, i) => {
    const hour = 7 + i;
    return `${hour}:00-${hour + 1}:00`;
  });

  const occupiedSlots = {};
  if (scheduleTable) {
    scheduleTable.innerHTML = "";
  }

  timeSlots.forEach((timeSlot, slotIndex) => {
    const row = document.createElement("tr");
    const timeCell = document.createElement("td");
    timeCell.textContent = timeSlot;
    timeCell.className = "time-slot";
    row.appendChild(timeCell);

    daysOfWeek.forEach((day) => {
      const slotKey = `${day}-${slotIndex}`;
      if (occupiedSlots[slotKey]) return;

      const cell = document.createElement("td");
      cell.className = "class-cell empty";

      const scheduleItem = data.find((item) => {
        if (!item?.time) {
          console.warn("Mục không có thời gian:", item);
          return false;
        }

        const parts = item.time.split(" ");
        if (parts.length < 3) {
          console.warn("Định dạng thời gian không hợp lệ:", item.time);
          return false;
        }

        const itemDay = parts.slice(0, -1).join(" ");
        const timeRange = parts[parts.length - 1];

        if (!timeRange.includes("-")) {
          console.warn("Khoảng thời gian không hợp lệ:", timeRange);
          return false;
        }

        const [startTime, endTime] = timeRange.split("-");
        const slotStartHour = parseInt(timeSlot.split("-")[0].split(":")[0]);
        const itemStartHour = parseInt(startTime.split(":")[0]);
        const itemEndHour = parseInt(endTime.split(":")[0]);

        return (
          normalizeDay(itemDay) === normalizeDay(day) &&
          slotStartHour >= itemStartHour &&
          slotStartHour < itemEndHour
        );
      });

      if (scheduleItem) {
        console.log("Tìm thấy môn học:", scheduleItem);
        const parts = scheduleItem.time.split(" ");
        const timeRange = parts[parts.length - 1];
        const [startTime, endTime] = timeRange.split("-");
        const startHour = parseInt(startTime.split(":")[0]);
        const endHour = parseInt(endTime.split(":")[0]);
        const rowspan = endHour - startHour;

        cell.className = `class-cell type-${scheduleItem.class_id % 5 || 1}`;
        cell.innerHTML = `
                    <div class="fw-semibold">${scheduleItem.subject}</div>
                    <div class="text-muted small">${scheduleItem.class}</div>
                    <div class="text-muted small mt-1">${
                      scheduleItem.time.split(" ")[1]
                    }</div>
                `;
        cell.setAttribute("data-class-id", scheduleItem.class_id);
        cell.setAttribute("data-timetable-id", scheduleItem.timetable_id);
        cell.setAttribute("data-subject", scheduleItem.subject);
        cell.addEventListener("click", () => handleScheduleClick(scheduleItem));

        if (rowspan > 1) {
          cell.setAttribute("rowspan", rowspan);
          for (let i = 1; i < rowspan; i++) {
            occupiedSlots[`${day}-${slotIndex + i}`] = true;
          }
        }
      }

      row.appendChild(cell);
    });

    scheduleTable && scheduleTable.appendChild(row);
  });
}

async function handleScheduleClick(item) {
  if (subjectName) subjectName.innerText = item.subject;
  if (className) className.innerText = item.class;
  currentClassId = item.class_id;
  currentTimetableId = item.timetable_id;
  showPage(pageAttendance);
  recognizedStudents.clear();
  studentMap.clear();
  if (studentList) studentList.innerHTML = "";
  await fetchStudents(item.class_id);
}

async function fetchStudents(classId) {
  try {
    const res = await fetch(`${BASE_URL}/api/class/${classId}/students`);
    const data = await res.json();
    if (studentList) {
      studentList.innerHTML = "";
      data.forEach((student) => {
        const tr = document.createElement("tr");
        tr.setAttribute("data-msv", student.MSV);
        tr.innerHTML = `
                    <td>${student.MSV}</td>
                    <td>${student.FullName}</td>
                    <td class="status"><input type="checkbox" class="status-checkbox" data-msv="${student.MSV}"></td>
                `;
        studentList.appendChild(tr);
        studentMap.set(student.MSV, tr);
      });

      document.querySelectorAll(".status-checkbox").forEach((checkbox) => {
        checkbox.addEventListener("change", (event) => {
          const msv = event.target.getAttribute("data-msv");
          if (event.target.checked) {
            recognizedStudents.add(msv);
          } else {
            recognizedStudents.delete(msv);
          }
        });
      });
    }
  } catch (error) {
    console.error("Error fetching students:", error);
    if (studentList) {
      studentList.innerHTML =
        "<tr><td colspan='3'>Lỗi khi tải danh sách sinh viên</td></tr>";
    }
  }
}

function startRecognitionPolling() {
  recognitionInterval = setInterval(async () => {
    if (
      pageAttendance &&
      !pageAttendance.classList.contains("hidden") &&
      videoContainer &&
      !videoContainer.classList.contains("hidden")
    ) {
      try {
        const res = await fetch(`${BASE_URL}/api/recognize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        const data = await res.json();
        const uniqueRecognized = [...new Set(data.recognized)];

        if (uniqueRecognized.length > 0) {
          console.log("Recognized MSVs:", uniqueRecognized);
        }

        if (data.recognized) {
          data.recognized.forEach((msv) => {
            if (studentMap.has(msv) && !recognizedStudents.has(msv)) {
              recognizedStudents.add(msv);
              const row = studentMap.get(msv);
              if (row) {
                const checkbox = row.querySelector(".status-checkbox");
                if (checkbox && !checkbox.checked) {
                  checkbox.checked = true;
                }
              }
            }
          });
        }
      } catch (error) {
        console.error("Error during recognition:", error);
      }
    } else {
      clearInterval(recognitionInterval);
      recognitionInterval = null;
      sidebar && sidebar.classList.remove("sidebar-collapsed");
    }
  }, 2000);
}

if (startFaceRecognitionBtn) {
  startFaceRecognitionBtn.addEventListener("click", async () => {
    if (!currentClassId) {
      if (statusIcon && statusMessage && statusModal) {
        statusIcon.innerHTML = `
                    <svg class="error-x" viewBox="0 0 52 52">
                        <path class="error-x" d="M16 16 36 36 M36 16 16 36" />
                    </svg>
                `;
        statusMessage.innerText =
          "Vui lòng chọn một lớp học trước khi bắt đầu nhận diện!";
        statusIcon.classList.remove("success");
        statusIcon.classList.add("error");
        statusModal.show();
      }
      return;
    }
    try {
      const res = await fetch(`${BASE_URL}/api/start_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_id: currentClassId }),
      });
      const data = await res.json();
      if (data.status === "success") {
        if (videoContainer) videoContainer.classList.remove("hidden");
        if (streamFrame) streamFrame.src = `${BASE_URL}/stream`;
        if (sidebar) sidebar.classList.add("sidebar-collapsed");
        startRecognitionPolling();
        if (statusIcon && statusMessage && statusModal) {
          statusIcon.innerHTML = `
                        <svg class="checkmark" viewBox="0 0 52 52">
                            <circle class="checkmark-circle" cx="26" cy="26" r="25" />
                            <path class="checkmark-check" d="M14.1 27.2l7.1 7.2 16.7-16.8" />
                        </svg>
                    `;
          statusMessage.innerText =
            data.message || "Bắt đầu luồng video thành công!";
          statusIcon.classList.remove("error");
          statusIcon.classList.add("success");
          statusModal.show();
        }
        if (stopFaceRecognitionBtn)
          stopFaceRecognitionBtn.classList.remove("hidden");
        startFaceRecognitionBtn.classList.add("hidden");
      } else {
        if (statusIcon && statusMessage && statusModal) {
          statusIcon.innerHTML = `
                        <svg class="error-x" viewBox="0 0 52 52">
                            <path class="error-x" d="M16 16 36 36 M36 16 16 36" />
                        </svg>
                    `;
          statusMessage.innerText =
            data.message || "Lỗi khi bắt đầu luồng video!";
          statusIcon.classList.remove("success");
          statusIcon.classList.add("error");
          statusModal.show();
        }
      }
    } catch (error) {
      console.error("Error starting stream:", error);
      if (statusIcon && statusMessage && statusModal) {
        statusIcon.innerHTML = `
                    <svg class="error-x" viewBox="0 0 52 52">
                        <path class="error-x" d="M16 16 36 36 M36 16 16 36" />
                    </svg>
                `;
        statusMessage.innerText = "Lỗi khi bắt đầu luồng video!";
        statusIcon.classList.remove("success");
        statusIcon.classList.add("error");
        statusModal.show();
      }
    }
  });
}

if (stopFaceRecognitionBtn) {
  stopFaceRecognitionBtn.addEventListener("click", async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/stop_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data.status === "success") {
        if (videoContainer) videoContainer.classList.add("hidden");
        if (streamFrame) streamFrame.src = "";
        if (sidebar) sidebar.classList.remove("sidebar-collapsed");
        if (recognitionInterval) {
          clearInterval(recognitionInterval);
          recognitionInterval = null;
        }
        if (statusIcon && statusMessage && statusModal) {
          statusIcon.innerHTML = `
                        <svg class="checkmark" viewBox="0 0 52 52">
                            <circle class="checkmark-circle" cx="26" cy="26" r="25" />
                            <path class="checkmark-check" d="M14.1 27.2l7.1 7.2 16.7-16.8" />
                        </svg>
                    `;
          statusMessage.innerText =
            data.message || "Đã dừng luồng video thành công!";
          statusIcon.classList.remove("error");
          statusIcon.classList.add("success");
          statusModal.show();
        }
        stopFaceRecognitionBtn.classList.add("hidden");
        if (startFaceRecognitionBtn)
          startFaceRecognitionBtn.classList.remove("hidden");
      } else {
        if (statusIcon && statusMessage && statusModal) {
          statusIcon.innerHTML = `
                        <svg class="error-x" viewBox="0 0 52 52">
                            <path class="error-x" d="M16 16 36 36 M36 16 16 36" />
                        </svg>
                    `;
          statusMessage.innerText = data.message || "Lỗi khi dừng luồng video!";
          statusIcon.classList.remove("success");
          statusIcon.classList.add("error");
          statusModal.show();
        }
      }
    } catch (error) {
      console.error("Error stopping stream:", error);
      if (statusIcon && statusMessage && statusModal) {
        statusIcon.innerHTML = `
                    <svg class="error-x" viewBox="0 0 52 52">
                        <path class="error-x" d="M16 16 36 36 M36 16 16 36" />
                    </svg>
                `;
        statusMessage.innerText = "Lỗi khi dừng luồng video!";
        statusIcon.classList.remove("success");
        statusIcon.classList.add("error");
        statusModal.show();
      }
    }
  });
}

const uploadForm = document.getElementById("uploadForm");
console.log("uploadForm found:", uploadForm);
if (uploadForm) {
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log("Form submit prevented");
    try {
      const code = document.getElementById("studentCode")?.value.trim();
      const files = document.getElementById("studentImage")?.files;
      const resultElement = document.getElementById("uploadResult");

      if (!code || !files || files.length === 0) {
        if (!code) {
          showErrorModal("Vui lòng nhập mã sinh viên!");
          if (resultElement)
            resultElement.innerText = "Vui lòng nhập mã sinh viên.";
        } else {
          showErrorModal("Vui lòng chọn ít nhất một ảnh!");
          if (resultElement)
            resultElement.innerText = "Vui lòng chọn ít nhất một ảnh.";
        }
        return;
      }

      const formData = new FormData();
      formData.append("student_name", code);
      for (let i = 0; i < files.length; i++) {
        formData.append("images[]", files[i]);
      }

      console.log("Sending fetch to:", `${BASE_URL}/api/upload`);
      const res = await fetch(`${BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      console.log("Response status:", res.status);
      const data = await res.json();

      if (resultElement) {
        resultElement.innerText = data.message || "Đã gửi ảnh.";
        resultElement.className =
          data.status === "success" ? "mt-3 text-success" : "mt-3 text-danger";
      }

      if (data.status === "success") {
        alert(data.message || "Đã gửi ảnh thành công!");
      } else {
        showErrorModal(data.message || "Lỗi khi gửi ảnh!");
      }
    } catch (error) {
      console.error("Error uploading image:", error);
      if (resultElement) {
        resultElement.innerText =
          "Lỗi khi gửi ảnh: " + (error.message || "Kết nối thất bại");
        resultElement.className = "mt-3 text-danger";
      }
      showErrorModal(
        "Lỗi khi gửi ảnh: " + (error.message || "Kết nối thất bại")
      );
    }
  });
} else {
  console.error("uploadForm not found!");
}

function showSuccessModal(message) {
  if (statusIcon && statusMessage && statusModal) {
    statusIcon.innerHTML = `
            <svg class="checkmark" viewBox="0 0 52 52">
                <circle class="checkmark-circle" cx="26" cy="26" r="25" />
                <path class="checkmark-check" d="M14.1 27.2l7.1 7.2 16.7-16.8" />
            </svg>
        `;
    statusMessage.innerText = message;
    statusIcon.classList.remove("error");
    statusIcon.classList.add("success");
    statusModal.show();
  }
}

function showErrorModal(message) {
  if (statusIcon && statusMessage && statusModal) {
    statusIcon.innerHTML = `
            <svg class="error-x" viewBox="0 0 52 52">
                <path class="error-x" d="M16 16 36 36 M36 16 16 36" />
            </svg>
        `;
    statusMessage.innerText = message;
    statusIcon.classList.remove("success");
    statusIcon.classList.add("error");
    statusModal.show();
  }
}

if (saveAttendanceBtn) {
  saveAttendanceBtn.addEventListener("click", async (event) => {
    event.preventDefault();
    if (isSaving || !currentTimetableId) {
      if (statusIcon && statusMessage && statusModal) {
        statusIcon.innerHTML = `
                    <svg class="error-x" viewBox="0 0 52 52">
                        <path class="error-x" d="M16 16 36 36 M36 16 16 36" />
                    </svg>
                `;
        statusMessage.innerText =
          "Vui lòng chọn một lớp học trước khi lưu điểm danh!";
        statusIcon.classList.remove("success");
        statusIcon.classList.add("error");
        statusModal.show();
      }
      return;
    }

    isSaving = true;

    const attendanceData = [];
    document.querySelectorAll("#studentList tr").forEach((row) => {
      const msv = row.getAttribute("data-msv");
      const name = row.children[1]?.innerText || "";
      const checkbox = row.querySelector(".status-checkbox");
      const status = checkbox && checkbox.checked ? "Present" : "Absent";
      attendanceData.push({ MSV: msv, FullName: name, Status: status });
    });

    console.log("Dữ liệu gửi lên:", {
      timetable_id: currentTimetableId,
      data: attendanceData,
    });

    try {
      const res = await fetch(`${BASE_URL}/api/save_attendance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          timetable_id: currentTimetableId,
          data: attendanceData,
        }),
      });
      const result = await res.json();
      if (statusIcon && statusMessage && statusModal) {
        if (result.status === "success") {
          showSuccessModal(result.message || "Đã lưu điểm danh thành công!");

          // Dừng nhận diện và ẩn video sau khi lưu thành công
          if (recognitionInterval) {
            clearInterval(recognitionInterval);
            recognitionInterval = null;
          }
          if (videoContainer) videoContainer.classList.add("hidden");
          if (streamFrame) streamFrame.src = "";
          if (sidebar) sidebar.classList.remove("sidebar-collapsed");
          if (startFaceRecognitionBtn)
            startFaceRecognitionBtn.classList.remove("hidden");
          if (stopFaceRecognitionBtn)
            stopFaceRecognitionBtn.classList.add("hidden");

          // Gửi yêu cầu dừng luồng video
          try {
            const stopRes = await fetch(`${BASE_URL}/api/stop_stream`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({}),
            });
            const stopData = await stopRes.json();
            if (stopData.status !== "success") {
              console.warn("Không thể dừng luồng video:", stopData.message);
            }
          } catch (stopError) {
            console.error("Lỗi khi dừng luồng video:", stopError);
          }
        } else {
          showErrorModal(result.message || "Lỗi khi lưu điểm danh!");
        }
      }
    } catch (error) {
      console.error("Error saving attendance:", error);
      if (statusIcon && statusMessage && statusModal) {
        showErrorModal("Lỗi khi lưu điểm danh: " + error.message);
      }
    } finally {
      isSaving = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  showPage(pageUpload);
});