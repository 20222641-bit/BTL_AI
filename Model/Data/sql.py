import sqlite3
from datetime import datetime

conn = sqlite3.connect("Model/Data/bien_so.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS tinh (
    MaTinh TEXT PRIMARY KEY,
    TenTinh TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS lichsu (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    BienSo TEXT NOT NULL,
    TenTinh TEXT NOT NULL,
    NgayGio DATETIME NOT NULL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS vexe (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,  
    BienSo TEXT NOT NULL                   
)
""")
tinh_data = [
    ('29', 'Hà Nội'),
    ('30', 'Hà Nội'),
    ('31', 'Hà Nội'),
    ('32', 'Hà Nội'),
    ('33', 'Hà Nội'),
    ('15', 'Hải Phòng'),
    ('16', 'Hải Phòng'),
    ('34', 'Hải Phòng'),
    ('43', 'Đà Nẵng'),
    ('92', 'Đà Nẵng'),
    ('73', 'Huế'),
    ('74', 'Huế'),
    ('75', 'Huế'),
    ('65', 'Cần Thơ'),
    ('83', 'Cần Thơ'),
    ('95', 'Cần Thơ'),
    ('50', 'TP Hồ Chí Minh'),
    ('51', 'TP Hồ Chí Minh'),
    ('52', 'TP Hồ Chí Minh'),
    ('53', 'TP Hồ Chí Minh'),
    ('54', 'TP Hồ Chí Minh'),
    ('55', 'TP Hồ Chí Minh'),
    ('56', 'TP Hồ Chí Minh'),
    ('57', 'TP Hồ Chí Minh'),
    ('58', 'TP Hồ Chí Minh'),
    ('59', 'TP Hồ Chí Minh'),
    ('61', 'TP Hồ Chí Minh'),
    ('72', 'TP Hồ Chí Minh'),
    ('11', 'Cao Bằng'),
    ('12', 'Lạng Sơn'),
    ('14', 'Quảng Ninh'),
    ('19', 'Phú Thọ'),
    ('88', 'Phú Thọ'),
    ('28', 'Phú Thọ'),
    ('20', 'Thái Nguyên'),
    ('97', 'Thái Nguyên'),
    ('21', 'Lào Cai'),
    ('24', 'Lào Cai'),
    ('22', 'Tuyên Quang'),
    ('23', 'Tuyên Quang'),
    ('17', 'Hưng Yên'),
    ('89', 'Hưng Yên'),
    ('35', 'Ninh Bình'),
    ('18', 'Ninh Bình'),
    ('90', 'Ninh Bình'),
    ('36', 'Thanh Hóa'),
    ('37', 'Nghệ An'),
    ('38', 'Hà Tĩnh'),
    ('76', 'Quảng Ngãi'),
    ('82', 'Quảng Ngãi'),
    ('77', 'Gia Lai'),
    ('81', 'Gia Lai'),
    ('47', 'Đắk Lắk'),
    ('78', 'Đắk Lắk'),
    ('79', 'Khánh Hòa'),
    ('85', 'Khánh Hòa'),
    ('49', 'Lâm Đồng'),
    ('48', 'Lâm Đồng'),
    ('86', 'Lâm Đồng'),
    ('60', 'Đồng Nai'),
    ('93', 'Đồng Nai'),
    ('70', 'Tây Ninh'),
    ('62', 'Tây Ninh'),
    ('66', 'Đồng Tháp'),
    ('63', 'Đồng Tháp'),
    ('64', 'Vĩnh Long'),
    ('71', 'Vĩnh Long'),
    ('84', 'Vĩnh Long'),
    ('67', 'An Giang'),
    ('68', 'An Giang'),
    ('69', 'Cà Mau'),
    ('94', 'Cà Mau'),
    ('25', 'Lai Châu'),
    ('27', 'Điện Biên'),
    ('26', 'Sơn La'),
]
for ma, ten in tinh_data:
    cursor.execute("INSERT OR IGNORE INTO tinh(MaTinh, TenTinh) VALUES(?, ?)", (ma, ten))
conn.commit()

def lay_tinh(bien_so):
    ma_tinh = bien_so[:2]  # 2 số đầu
    cursor.execute("SELECT TenTinh FROM tinh WHERE MaTinh=?", (ma_tinh,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return "Không xác định"

def luu_lich_su(bien_so):
    ten_tinh = lay_tinh(bien_so)
    ngay_gio = datetime.now()
    cursor.execute("INSERT INTO lichsu(BienSo, TenTinh, NgayGio) VALUES(?, ?, ?)",
                   (bien_so, ten_tinh, ngay_gio))
    conn.commit()
    return ten_tinh

def add_ticket(bien_so: str):
    cur = conn.cursor()
    cur.execute("INSERT INTO vexe (BienSo) VALUES (?)", (bien_so,))
    conn.commit()
    conn.close()

def list_tickets():
    cur = conn.cursor()
    cur.execute("SELECT ID, BienSo FROM vexe")
    rows = cur.fetchall()
    conn.close()
    return rows
for row in list_tickets():
    print(row)