// crispasr_async_jobs.h — async transcription job queue backed by SQLite.
//
// Components:
//   CrispasrJobStore   — SQLite CRUD for job records
//   CrispasrJobWorker  — background thread pool that processes queued jobs
//   crispasr_async_cleanup_start/stop — periodic cleanup of expired jobs
//
// Usage in crispasr_server.cpp:
//   1. After backend init: create JobStore, start JobWorker + cleanup thread
//   2. POST handler: if async=true, save audio + enqueue, return 202
//   3. GET handler: poll job_store.get_job(id)
//   4. DELETE handler: cancel/delete via job_store
//   5. Shutdown: worker.stop(), cleanup_stop()

#pragma once

#include "crispasr_backend.h"
#include "crispasr_output.h"
#include "whisper_params.h"
#include "fireredpunc.h"
#include "pcs.h"

#include <sqlite3.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Forward declarations from crispasr_server.cpp (shared helpers).
struct transcription_result;
bool read_audio_data(const std::string& fname, std::vector<float>& pcmf32,
                     std::vector<std::vector<float>>& pcmf32s, bool stereo);

// ---------------------------------------------------------------------------
// Job record
// ---------------------------------------------------------------------------

struct CrispasrJob {
    std::string id;
    std::string status;      // queued | processing | completed | failed
    int64_t     created_at = 0;
    int64_t     started_at = 0;
    int64_t     completed_at = 0;
    std::string audio_path;
    std::string request_params;  // JSON
    std::string response_format;
    std::string result;          // JSON
    std::string error;
    double      progress = 0.0;
};

// ---------------------------------------------------------------------------
// Job ID generation
// ---------------------------------------------------------------------------

inline std::string crispasr_generate_job_id() {
    static std::mutex rng_mtx;
    static std::mt19937_64 rng([] {
        std::random_device rd;
        return rd();
    }());
    std::lock_guard<std::mutex> lock(rng_mtx);
    uint64_t val = rng();
    char buf[32];
    snprintf(buf, sizeof(buf), "job_%012llx", (unsigned long long)(val & 0xFFFFFFFFFFFFULL));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// CrispasrJobStore — SQLite persistence
// ---------------------------------------------------------------------------

class CrispasrJobStore {
public:
    CrispasrJobStore() = default;
    ~CrispasrJobStore() { close(); }

    CrispasrJobStore(const CrispasrJobStore&) = delete;
    CrispasrJobStore& operator=(const CrispasrJobStore&) = delete;

    bool open(const std::string& db_path) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (db_) return true;
        int rc = sqlite3_open(db_path.c_str(), &db_);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "crispasr-async: failed to open DB '%s': %s\n",
                    db_path.c_str(), sqlite3_errmsg(db_));
            db_ = nullptr;
            return false;
        }
        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA synchronous=NORMAL");
        exec("CREATE TABLE IF NOT EXISTS jobs ("
             "id              TEXT PRIMARY KEY,"
             "status          TEXT NOT NULL DEFAULT 'queued',"
             "created_at      INTEGER NOT NULL,"
             "started_at      INTEGER,"
             "completed_at    INTEGER,"
             "audio_path      TEXT NOT NULL,"
             "request_params  TEXT NOT NULL,"
             "response_format TEXT NOT NULL DEFAULT 'json',"
             "result          TEXT,"
             "error           TEXT,"
             "progress        REAL NOT NULL DEFAULT 0.0)");
        exec("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)");
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

    bool create_job(const std::string& id, const std::string& audio_path,
                    const std::string& params_json, const std::string& response_format) {
        std::lock_guard<std::mutex> lock(mtx_);
        const char* sql = "INSERT INTO jobs (id, status, created_at, audio_path, request_params, response_format) "
                          "VALUES (?, 'queued', ?, ?, ?, ?)";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        int64_t now = epoch_seconds();
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, now);
        sqlite3_bind_text(stmt, 3, audio_path.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, params_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, response_format.c_str(), -1, SQLITE_TRANSIENT);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    CrispasrJob get_job(const std::string& id) {
        std::lock_guard<std::mutex> lock(mtx_);
        CrispasrJob job;
        const char* sql = "SELECT id, status, created_at, started_at, completed_at, "
                          "audio_path, request_params, response_format, result, error, progress "
                          "FROM jobs WHERE id = ?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return job;
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            job = row_to_job(stmt);
        }
        sqlite3_finalize(stmt);
        return job;
    }

    CrispasrJob claim_next() {
        std::lock_guard<std::mutex> lock(mtx_);
        CrispasrJob job;
        int64_t now = epoch_seconds();
        const char* sql = "UPDATE jobs SET status='processing', started_at=? "
                          "WHERE id = (SELECT id FROM jobs WHERE status='queued' "
                          "ORDER BY created_at LIMIT 1) RETURNING "
                          "id, status, created_at, started_at, completed_at, "
                          "audio_path, request_params, response_format, result, error, progress";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return job;
        sqlite3_bind_int64(stmt, 1, now);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            job = row_to_job(stmt);
        }
        sqlite3_finalize(stmt);
        return job;
    }

    bool update_progress(const std::string& id, double progress) {
        std::lock_guard<std::mutex> lock(mtx_);
        const char* sql = "UPDATE jobs SET progress=? WHERE id=?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_double(stmt, 1, progress);
        sqlite3_bind_text(stmt, 2, id.c_str(), -1, SQLITE_TRANSIENT);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool complete_job(const std::string& id, const std::string& result_json) {
        std::lock_guard<std::mutex> lock(mtx_);
        const char* sql = "UPDATE jobs SET status='completed', completed_at=?, progress=1.0, result=? WHERE id=?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_int64(stmt, 1, epoch_seconds());
        sqlite3_bind_text(stmt, 2, result_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, id.c_str(), -1, SQLITE_TRANSIENT);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool fail_job(const std::string& id, const std::string& error) {
        std::lock_guard<std::mutex> lock(mtx_);
        const char* sql = "UPDATE jobs SET status='failed', completed_at=?, error=? WHERE id=?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_int64(stmt, 1, epoch_seconds());
        sqlite3_bind_text(stmt, 2, error.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, id.c_str(), -1, SQLITE_TRANSIENT);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool delete_job(const std::string& id, std::string& audio_path_out) {
        std::lock_guard<std::mutex> lock(mtx_);
        const char* sql_get = "SELECT audio_path FROM jobs WHERE id=?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql_get, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                const char* p = (const char*)sqlite3_column_text(stmt, 0);
                if (p) audio_path_out = p;
            }
            sqlite3_finalize(stmt);
        }
        const char* sql_del = "DELETE FROM jobs WHERE id=?";
        if (sqlite3_prepare_v2(db_, sql_del, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE && sqlite3_changes(db_) > 0;
        sqlite3_finalize(stmt);
        return ok;
    }

    int count_pending() {
        std::lock_guard<std::mutex> lock(mtx_);
        const char* sql = "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')";
        sqlite3_stmt* stmt = nullptr;
        int count = 0;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW)
                count = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
        }
        return count;
    }

    void recover_dangling() {
        std::lock_guard<std::mutex> lock(mtx_);
        // Find all processing jobs — re-queue if audio exists, fail otherwise.
        const char* sql = "SELECT id, audio_path FROM jobs WHERE status='processing'";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
        std::vector<std::pair<std::string, std::string>> dangling;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* id = (const char*)sqlite3_column_text(stmt, 0);
            const char* path = (const char*)sqlite3_column_text(stmt, 1);
            if (id) dangling.emplace_back(id, path ? path : "");
        }
        sqlite3_finalize(stmt);

        for (auto& [id, path] : dangling) {
            std::error_code ec;
            if (!path.empty() && std::filesystem::exists(path, ec)) {
                exec_bind("UPDATE jobs SET status='queued', started_at=NULL, progress=0.0 WHERE id=?", id);
                fprintf(stderr, "crispasr-async: recovered dangling job %s -> re-queued\n", id.c_str());
            } else {
                exec_bind("UPDATE jobs SET status='failed', completed_at=?, error='server restarted, audio lost' WHERE id=?",
                          epoch_seconds(), id);
                fprintf(stderr, "crispasr-async: recovered dangling job %s -> failed (audio missing)\n", id.c_str());
            }
        }
    }

    int cleanup_expired(int64_t max_age_seconds) {
        std::lock_guard<std::mutex> lock(mtx_);
        int64_t cutoff = epoch_seconds() - max_age_seconds;

        // Collect audio paths before deleting.
        const char* sql_sel = "SELECT audio_path FROM jobs WHERE status IN ('completed','failed') AND completed_at < ?";
        sqlite3_stmt* stmt = nullptr;
        std::vector<std::string> paths;
        if (sqlite3_prepare_v2(db_, sql_sel, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(stmt, 1, cutoff);
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                const char* p = (const char*)sqlite3_column_text(stmt, 0);
                if (p && *p) paths.emplace_back(p);
            }
            sqlite3_finalize(stmt);
        }

        // Delete rows.
        const char* sql_del = "DELETE FROM jobs WHERE status IN ('completed','failed') AND completed_at < ?";
        if (sqlite3_prepare_v2(db_, sql_del, -1, &stmt, nullptr) != SQLITE_OK) return 0;
        sqlite3_bind_int64(stmt, 1, cutoff);
        sqlite3_step(stmt);
        int deleted = sqlite3_changes(db_);
        sqlite3_finalize(stmt);

        // Remove audio files.
        for (auto& p : paths) {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
        return deleted;
    }

private:
    sqlite3* db_ = nullptr;
    std::mutex mtx_;

    static int64_t epoch_seconds() {
        return (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void exec(const char* sql) {
        char* err = nullptr;
        sqlite3_exec(db_, sql, nullptr, nullptr, &err);
        if (err) {
            fprintf(stderr, "crispasr-async: SQL error: %s\n", err);
            sqlite3_free(err);
        }
    }

    void exec_bind(const char* sql, const std::string& text_val) {
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, text_val.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    void exec_bind(const char* sql, int64_t int_val, const std::string& text_val) {
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(stmt, 1, int_val);
            sqlite3_bind_text(stmt, 2, text_val.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    static CrispasrJob row_to_job(sqlite3_stmt* stmt) {
        CrispasrJob j;
        auto text_col = [&](int col) -> std::string {
            const char* p = (const char*)sqlite3_column_text(stmt, col);
            return p ? p : "";
        };
        j.id              = text_col(0);
        j.status          = text_col(1);
        j.created_at      = sqlite3_column_int64(stmt, 2);
        j.started_at      = sqlite3_column_int64(stmt, 3);
        j.completed_at    = sqlite3_column_int64(stmt, 4);
        j.audio_path      = text_col(5);
        j.request_params  = text_col(6);
        j.response_format = text_col(7);
        j.result          = text_col(8);
        j.error           = text_col(9);
        j.progress        = sqlite3_column_double(stmt, 10);
        return j;
    }
};

// ---------------------------------------------------------------------------
// CrispasrJobWorker — background worker pool
// ---------------------------------------------------------------------------

class CrispasrJobWorker {
public:
    using TranscribeFn = std::function<transcription_result(const std::string& audio_path,
                                                            const std::string& params_json,
                                                            const std::string& job_id)>;

    CrispasrJobWorker() = default;
    ~CrispasrJobWorker() { stop(); }

    CrispasrJobWorker(const CrispasrJobWorker&) = delete;
    CrispasrJobWorker& operator=(const CrispasrJobWorker&) = delete;

    void start(int n_workers, CrispasrJobStore& store, TranscribeFn fn) {
        store_ = &store;
        transcribe_fn_ = std::move(fn);
        shutdown_.store(false);
        for (int i = 0; i < n_workers; ++i) {
            workers_.emplace_back(&CrispasrJobWorker::worker_loop, this, i);
        }
        fprintf(stderr, "crispasr-async: started %d worker thread(s)\n", n_workers);
    }

    void stop() {
        shutdown_.store(true);
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
    }

    void notify() {
        cv_.notify_one();
    }

    void request_cancel(const std::string& job_id) {
        std::lock_guard<std::mutex> lock(cancel_mtx_);
        cancel_flags_[job_id].store(true);
    }

    bool is_cancelled(const std::string& job_id) {
        std::lock_guard<std::mutex> lock(cancel_mtx_);
        auto it = cancel_flags_.find(job_id);
        return it != cancel_flags_.end() && it->second.load();
    }

    void clear_cancel(const std::string& job_id) {
        std::lock_guard<std::mutex> lock(cancel_mtx_);
        cancel_flags_.erase(job_id);
    }

private:
    CrispasrJobStore* store_ = nullptr;
    TranscribeFn transcribe_fn_;
    std::vector<std::thread> workers_;
    std::atomic<bool> shutdown_{false};
    std::mutex wait_mtx_;
    std::condition_variable cv_;
    std::mutex cancel_mtx_;
    std::unordered_map<std::string, std::atomic<bool>> cancel_flags_;

    void worker_loop(int worker_id) {
        fprintf(stderr, "crispasr-async: worker %d started\n", worker_id);
        while (!shutdown_.load()) {
            CrispasrJob job = store_->claim_next();
            if (job.id.empty()) {
                std::unique_lock<std::mutex> lock(wait_mtx_);
                cv_.wait_for(lock, std::chrono::seconds(2), [this] {
                    return shutdown_.load();
                });
                continue;
            }

            fprintf(stderr, "crispasr-async: worker %d processing job %s\n", worker_id, job.id.c_str());

            if (is_cancelled(job.id)) {
                store_->fail_job(job.id, "cancelled");
                remove_audio(job.audio_path);
                clear_cancel(job.id);
                continue;
            }

            try {
                auto result = transcribe_fn_(job.audio_path, job.request_params, job.id);
                if (is_cancelled(job.id)) {
                    store_->fail_job(job.id, "cancelled");
                    remove_audio(job.audio_path);
                    clear_cancel(job.id);
                    continue;
                }

                if (!result.ok) {
                    store_->fail_job(job.id, result.error);
                } else {
                    std::string result_json = format_result_json(result);
                    store_->complete_job(job.id, result_json);
                }
            } catch (const std::exception& e) {
                store_->fail_job(job.id, std::string("exception: ") + e.what());
            } catch (...) {
                store_->fail_job(job.id, "unknown exception");
            }

            remove_audio(job.audio_path);
            clear_cancel(job.id);
            fprintf(stderr, "crispasr-async: worker %d finished job %s\n", worker_id, job.id.c_str());
        }
        fprintf(stderr, "crispasr-async: worker %d stopped\n", worker_id);
    }

    static void remove_audio(const std::string& path) {
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    static std::string format_result_json(const transcription_result& r) {
        // Store segments + metadata as JSON for later formatting.
        // Use the same JSON builder as the server's verbose_json output.
        std::string task = "transcribe";
        std::string json = crispasr_segments_to_openai_verbose_json(
            r.segs, r.duration_s, r.language, task, 0.0f);
        return json;
    }
};

// ---------------------------------------------------------------------------
// Cleanup thread
// ---------------------------------------------------------------------------

inline void crispasr_async_cleanup_start(CrispasrJobStore& store, std::thread& thread,
                                          std::atomic<bool>& shutdown, int64_t max_age_seconds = 86400) {
    shutdown.store(false);
    thread = std::thread([&store, &shutdown, max_age_seconds] {
        // Run recovery on startup.
        store.recover_dangling();

        while (!shutdown.load()) {
            for (int i = 0; i < 60 && !shutdown.load(); ++i)
                std::this_thread::sleep_for(std::chrono::seconds(10));
            if (shutdown.load()) break;
            int n = store.cleanup_expired(max_age_seconds);
            if (n > 0)
                fprintf(stderr, "crispasr-async: cleaned up %d expired job(s)\n", n);
        }
    });
}

inline void crispasr_async_cleanup_stop(std::thread& thread, std::atomic<bool>& shutdown) {
    shutdown.store(true);
    if (thread.joinable()) thread.join();
}
