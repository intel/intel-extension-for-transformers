from config.config import get_settings
from pymysql import connect, cursors
from contextlib import contextmanager

global_settings = get_settings()

class MysqlDb(object):
    def __init__(self):
        self._host = global_settings.mysql_host
        self._port = global_settings.mysql_port
        self._db = global_settings.mysql_db
        self._user = global_settings.mysql_user
        self._passwd = global_settings.mysql_password
        self._charset = 'utf8'
        self._connect()

    def _connect(self):
        self._conn = connect(host=self._host,
                             port=self._port,
                             user=self._user,
                             passwd=self._passwd,
                             db=self._db,
                             charset=self._charset,
                             cursorclass=cursors.DictCursor)
        self._cursor = self._conn.cursor()

    def _close(self):
        self._cursor.close()
        self._conn.close()

    @contextmanager
    def transaction(self):
        try:
            yield
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e

    def fetch_one(self, sql, params=None):
        self._cursor.execute(sql, params)
        return self._cursor.fetchone()

    def fetch_all(self, sql, params=None):
        self._cursor.execute(sql, params)
        return self._cursor.fetchall()

    def insert(self, sql, params):
        return self._edit(sql, params)

    def update(self, sql, params):
        return self._edit(sql, params)

    def delete(self, sql, params):
        return self._edit(sql, params)

    def _edit(self, sql, params):
        return self._cursor.execute(sql, params)
