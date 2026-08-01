package com.example.halluciguard.db

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import com.example.halluciguard.model.AuditLogEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface AuditLogDao {

    @Query("SELECT * FROM audit_logs ORDER BY timestamp DESC")
    fun getAllAuditLogs(): Flow<List<AuditLogEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAuditLog(log: AuditLogEntity): Long

    @Query("DELETE FROM audit_logs")
    suspend fun clearAllAuditLogs()

    @Query("SELECT * FROM audit_logs WHERE id = :id")
    suspend fun getAuditLogById(id: Long): AuditLogEntity?
}

@Database(entities = [AuditLogEntity::class], version = 2, exportSchema = false)
abstract class AuditDatabase : RoomDatabase() {
    abstract fun auditLogDao(): AuditLogDao

    companion object {
        @Volatile
        private var INSTANCE: AuditDatabase? = null

        fun getDatabase(context: Context): AuditDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AuditDatabase::class.java,
                    "halluciguard_audit.db"
                )
                .fallbackToDestructiveMigration()
                .build()
                INSTANCE = instance
                instance
            }
        }
    }
}

class AuditRepository(private val dao: AuditLogDao) {
    val allLogs: Flow<List<AuditLogEntity>> = dao.getAllAuditLogs()

    suspend fun saveAuditLog(log: AuditLogEntity) = dao.insertAuditLog(log)
    suspend fun clearLogs() = dao.clearAllAuditLogs()
}
