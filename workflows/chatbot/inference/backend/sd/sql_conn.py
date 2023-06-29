import pymysql
from threading import Condition
import threading
import time
from config import SQL_HOST, SQL_PORT, SQL_USER, SQL_PASSWORD, SQL_DB, SQL_TABLE
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("mysql")

time_to_sleep = 1000

class MysqlWorker():
    def __init__(self):
        # threading.Thread.__init__(self)

        self.conn = pymysql.connect(host=SQL_HOST,port=SQL_PORT,user=SQL_USER,passwd=SQL_PASSWORD,db=SQL_DB)
        self.cursor = self.conn.cursor()

        self.lock = threading.Lock()

        self.save_sql = "insert into {} (user,timestamp,prompt,num_inference_steps,guidance_scale,seed,status,task_type,ip,port) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)".format(SQL_TABLE)
        self.update_sql = "UPDATE {} SET status=%s,execution_time=%s WHERE user=%s and timestamp=%s;".format(SQL_TABLE)

        self.completed_sql = "select max(id) as id from {} where status='completed'".format(SQL_TABLE)
        self.waiting_sql = "select id from {} where user='{}' and status='waiting'"

    def re_connection(self):
        try:
            self.cursor.close()
            self.conn.close()
        except:
            LOGGER.info("can't close cursor/conn.")

        LOGGER.info("re_connection.")

        self.conn = pymysql.connect(host=SQL_HOST,port=SQL_PORT,user=SQL_USER,passwd=SQL_PASSWORD,db=SQL_DB)
        self.cursor = self.conn.cursor()

    def run(self):
        # check connection status
        while True:
            time.sleep(time_to_sleep)
            """
            LOGGER.info("ping mysql.")
            self.conn.ping(reconnect=True)
            
            if not self._ping():
                self.re_connection()
            """

    def _ping(self):
        try:
            self.cursor.execute('select 1;')
            LOGGER.debug(self.cursor.fetchall())
            return True

        except pymysql.OperationalError as e:
            LOGGER.warn('Cannot connect to mysql - retrying in {} seconds'.format(self.time_to_sleep))
            LOGGER.exception(e)
            return False

    def query(self, value):
        print("query..................")
        print(value)
        self.lock.acquire()
        self.cursor.execute(self.completed_sql)
        latest_completed_id = self.cursor.fetchall()[0][0]
        self.cursor.execute(self.waiting_sql.format(SQL_TABLE, value))
        user_waiting_id = self.cursor.fetchall()[0][0]
        self.lock.release()
        print(latest_completed_id)
        print(user_waiting_id)
        return {"queue": user_waiting_id - latest_completed_id}

    def update(self, *values):
        LOGGER.info("update status to sql")
        self.lock.acquire()
        result = None
        try:
            result = self.cursor.execute(self.update_sql, values)
            self.conn.commit()
        except pymysql.OperationalError as e:
            LOGGER.info('Cannot update to mysql, will ping reconnect')
            LOGGER.exception(e)
            self.conn.ping(reconnect=True)
            self.cursor = self.conn.cursor()
            result = self.cursor.execute(self.update_sql, values)
            self.conn.commit()
        finally:
            self.lock.release()
        
        return result


    def insert(self, *values):
        LOGGER.info("save to sql")
        args = [values]
        self.lock.acquire()
        result = None
        print(args)
        try:
            result = self.cursor.executemany(query=self.save_sql, args=args)
            self.conn.commit()
        except pymysql.OperationalError as e:
            LOGGER.info('Cannot insert to mysql, will ping reconnect')
            LOGGER.exception(e)
            self.conn.ping(reconnect=True)
            self.cursor = self.conn.cursor()
            result = self.cursor.executemany(query=self.save_sql, args=args)
            self.conn.commit()
        finally:
            self.lock.release()
        
        return result

mysql = MysqlWorker()
# mysql.start()
