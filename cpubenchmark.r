# ============================================================
# cpubenchmark.r — Base R only (Win/Linux/macOS/WSL)
# - source("bench_solve.R") 하면:
#   1) 인터랙티브면 멀티스레드 on/off 질의
#   2) 벤치마크 즉시 실행 (여러 행렬 크기 × 방법 조합)
#   3) 결과 콘솔 출력 + CSV 자동 저장
# - 비인터랙티브(Rscript)면 프롬프트 없이 기본값으로 동작
# ============================================================

## ===== 사용자 조정 기본값 =====
.BENCH_SIZES   <- c(256L, 512L, 1024L, 2048L)   # 행렬 크기들
.BENCH_REPS    <- 5L                             # 반복 측정
.BENCH_WARMUP  <- 1L                             # 워밍업
.SEED_BASE     <- 2025L                          # 시드
.SAVE_CSV      <- TRUE                           # CSV 저장 여부

## ===== 공용 유틸 =====
.env_info <- function() {
  si <- Sys.info(); rv <- R.Version()
  esv <- tryCatch(utils::extSoftVersion(), error = function(e) list())
  blas <- unname(esv[["BLAS"]]); if (is.null(blas)) blas <- ""
  lapack <- unname(esv[["LAPACK"]]); if (is.null(lapack)) lapack <- ""
  cores <- tryCatch(parallel::detectCores(logical = TRUE), error = function(e) NA_integer_)
  data.frame(
    os       = paste0(si[["sysname"]], " ", si[["release"]]),
    machine  = si[["nodename"]],
    r_ver    = paste0(rv$major, ".", rv$minor),
    platform = rv$platform,
    BLAS     = blas, LAPACK = lapack,
    cores_logical = cores,
    stringsAsFactors = FALSE
  )
}

.get_env_threads <- function() {
  c(
    MKL_NUM_THREADS         = Sys.getenv("MKL_NUM_THREADS", ""),
    OPENBLAS_NUM_THREADS    = Sys.getenv("OPENBLAS_NUM_THREADS", ""),
    OMP_NUM_THREADS         = Sys.getenv("OMP_NUM_THREADS", ""),
    VECLIB_MAXIMUM_THREADS  = Sys.getenv("VECLIB_MAXIMUM_THREADS", ""),
    BLIS_NUM_THREADS        = Sys.getenv("BLIS_NUM_THREADS", "")
  )
}

.apply_threads <- function(enable_multithread = TRUE) {
  n_all <- tryCatch(parallel::detectCores(logical = TRUE), error = function(e) NA_integer_)
  if (isTRUE(enable_multithread)) {
    n <- if (is.na(n_all)) 0L else n_all
    if (n > 0L) {
      Sys.setenv(
        MKL_NUM_THREADS = n,
        OPENBLAS_NUM_THREADS = n,
        OMP_NUM_THREADS = n,
        VECLIB_MAXIMUM_THREADS = n,
        BLIS_NUM_THREADS = n
      )
      return(n)
    } else {
      Sys.unsetenv(c("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"))
      return("auto")
    }
  } else {
    Sys.setenv(
      MKL_NUM_THREADS = 1L,
      OPENBLAS_NUM_THREADS = 1L,
      OMP_NUM_THREADS = 1L,
      VECLIB_MAXIMUM_THREADS = 1L,
      BLIS_NUM_THREADS = 1L
    )
    return(1L)
  }
}

.make_matrix <- function(n, type = c("general","spd")) {
  type <- match.arg(type)
  if (type == "general") {
    A <- matrix(rnorm(n * n), n, n); diag(A) <- diag(A) + 1e-8; A
  } else {
    M <- matrix(rnorm(n * n), n, n); A <- crossprod(M); diag(A) <- diag(A) + 1e-8; A
  }
}

.solve_once <- function(A, b, method = c("solve","chol")) {
  method <- match.arg(method)
  if (method == "solve") {
    invisible(solve(A, b))                # LU (LAPACK)
  } else {
    R <- chol(A); y <- forwardsolve(t(R), b, upper.tri=TRUE, transpose=FALSE)
    x <- backsolve(R, y, upper.tri=TRUE, transpose=FALSE); invisible(x)
  }
}

.time_expr <- function(expr) {
  gc()
  as.numeric(system.time(eval.parent(substitute(expr)))[["elapsed"]])
}

.bench_one <- function(n, reps, warmup, type, method, seed) {
  set.seed(seed + n)
  A <- .make_matrix(n, type); b <- rnorm(n)
  if (method == "chol" && type != "spd") warning("chol을 SPD가 아닌 행렬에 사용 중")
  if (warmup > 0L) for (i in seq_len(warmup)) .solve_once(A, b, method)
  ts <- numeric(reps)
  for (i in seq_len(reps)) ts[i] <- .time_expr(.solve_once(A, b, method))
  flops <- if (method == "chol") (1/3) * n^3 else (2/3) * n^3
  gflops <- flops / ts / 1e9
  data.frame(
    type   = type, method = method, n = n, reps = reps, warmup = warmup,
    time_min_s = min(ts), time_median_s = stats::median(ts),
    time_mean_s = mean(ts), time_sd_s = stats::sd(ts), time_max_s = max(ts),
    gflops_min = min(gflops), gflops_median = stats::median(gflops),
    gflops_mean = mean(gflops), gflops_max = max(gflops),
    stringsAsFactors = FALSE
  )
}

run_bench <- function(sizes = .BENCH_SIZES,
                      reps = .BENCH_REPS,
                      warmup = .BENCH_WARMUP,
                      seed = .SEED_BASE,
                      multithread = TRUE) {
  applied <- .apply_threads(multithread)
  info    <- .env_info()
  envthr  <- .get_env_threads()
  stamp   <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
  
  combos <- list(
    list(type="general", method="solve"),
    list(type="spd",     method="chol")
  )
  res <- do.call(rbind, lapply(combos, function(cmb) {
    do.call(rbind, lapply(sizes, function(n) {
      .bench_one(n, reps, warmup, cmb$type, cmb$method, seed)
    }))
  }))
  
  res$timestamp <- stamp
  res$os        <- info$os
  res$machine   <- info$machine
  res$r_ver     <- info$r_ver
  res$platform  <- info$platform
  res$BLAS      <- info$BLAS
  res$LAPACK    <- info$LAPACK
  res$cores_logical <- info$cores_logical
  res$threads_mode  <- if (identical(applied, 1L)) "single" else "multi"
  res$threads_applied <- if (is.numeric(applied)) applied else as.character(applied)
  res$OMP_NUM_THREADS         <- envthr[["OMP_NUM_THREADS"]]
  res$OPENBLAS_NUM_THREADS    <- envthr[["OPENBLAS_NUM_THREADS"]]
  res$MKL_NUM_THREADS         <- envthr[["MKL_NUM_THREADS"]]
  res$VECLIB_MAXIMUM_THREADS  <- envthr[["VECLIB_MAXIMUM_THREADS"]]
  res$BLIS_NUM_THREADS        <- envthr[["BLIS_NUM_THREADS"]]
  
  res <- res[, c(
    "timestamp","threads_mode","threads_applied",
    "os","machine","r_ver","platform","BLAS","LAPACK","cores_logical",
    "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS",
    "type","method","n","reps","warmup",
    "time_min_s","time_median_s","time_mean_s","time_sd_s","time_max_s",
    "gflops_min","gflops_median","gflops_mean","gflops_max"
  )]
  
  if (isTRUE(.SAVE_CSV)) {
    fn <- sprintf("bench_solve_%s_%s.csv", res$threads_mode[1], stamp)
    utils::write.csv(res, fn, row.names = FALSE)
    message(">> Saved: ", fn)
  }
  res
}

## ===== 인터랙티브 진입점 =====
bench_solve_interactive <- function() {
  is_inter <- interactive()
  use_multi <- TRUE
  if (is_inter) {
    ans <- tryCatch(readline("Use multithreading? (y/n, default: y) : "), error=function(e) "")
    ans <- tolower(trimws(ans))
    if (nzchar(ans)) use_multi <- ans %in% c("y","yes","1","t","true")
  }
  cat(sprintf("\n[bench] multithreading = %s\n", if (use_multi) "ON" else "OFF"))
  cat(sprintf("[bench] sizes = %s, reps = %d, warmup = %d\n\n",
              paste(.BENCH_SIZES, collapse=","), .BENCH_REPS, .BENCH_WARMUP))
  res <- run_bench(sizes = .BENCH_SIZES,
                   reps = .BENCH_REPS,
                   warmup = .BENCH_WARMUP,
                   seed = .SEED_BASE,
                   multithread = use_multi)
  print(res, row.names = FALSE)
  invisible(res)
}

## ===== Rscript용 메인 (비인터랙티브) =====
bench_solve_main <- function() {
  # 비인터랙티브 기본: multithread = TRUE
  run_bench(multithread = TRUE)
}

## ===== 자동 실행 트리거 =====
if (sys.nframe() == 0L) {
  # source()로 불리면 인터랙티브 진입, Rscript면 non-interactive
  if (interactive()) {
    bench_solve_interactive()
  } else {
    bench_solve_main()
  }
}
# =========================== EOF ===========================
