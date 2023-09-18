from ..config.config import get_settings
from pymysql import connect

global_settings = get_settings()

class MysqlDb(object):
    def __init__(self):
        self._host = global_settings.mysql_host
        self._port = global_settings.mysql_port
        self._db = global_settings.mysql_db
        self._user = global_settings.mysql_user
        self._passwd = global_settings.mysql_password
        self._charset = 'utf8'

    def _connect(self):
        self._conn = connect(host=self._host,
                             port=self._port,
                             user=self._user,
                             passwd=self._passwd,
                             db=self._db,
                             charset=self._charset)
        self._cursor = self._conn.cursor()

    def _close(self):
        self._cursor.close()
        self._conn.close()

# =================== ADD =======================
    def set_db(self, db):
        self._db = db

    def fetch_one(self, sql, params=None):
        result = None
        try:
            self._connect()
            self._cursor.execute(sql, params)
            result = self._cursor.fetchone()
            self._close()
        except Exception as e:
            raise Exception(e)
        return result

    def fetch_all(self, sql, params=None):
        lst = ()
        try:
            self._connect()
            self._cursor.execute(sql, params)
            lst = self._cursor.fetchall()
            self._close()
        except Exception as e:
            raise Exception(e)
        return lst

    def insert(self, sql, params):
        return self._edit(sql, params)

    def update(self, sql, params):
        return self._edit(sql, params)

    def delete(self, sql, params):
        return self._edit(sql, params)

    def _edit(self, sql, params):
        count = 0
        try:
            self._connect()
            count = self._cursor.execute(sql, params)
            self._conn.commit()
            self._close()
        except Exception as e:
            raise Exception(e)
        return count
