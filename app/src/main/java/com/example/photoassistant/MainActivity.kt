
package com.example.photoassistant

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.os.Environment
import android.os.Build
import android.graphics.ImageFormat
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView

import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.compose.LocalLifecycleOwner

import java.util.Locale
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB

private const val TAG = "PhotoAssistant"
private const val PREFS_NAME = "PhotoAssistantPrefs"

private const val ORB_FEATURE_COUNT = 500
private const val RATIO_THRESHOLD = 0.75

// Initial experimental thresholds; calibrate them with real captures.
private const val MIN_SHARPNESS = 35.0
private const val MIN_GOOD_MATCHES = 15
private const val MIN_INLIERS = 10
private const val MIN_INLIER_RATIO = 0.30
private const val MIN_MOVEMENT_PIXELS = 45.0
private const val CAPTURE_COOLDOWN_MS = 3000L
private const val DUPLICATE_VIEW_MOVEMENT_PIXELS = 40.0
private const val REVERSE_WARNING_DEGREES = 18.0
private const val LOOP_COMPLETION_DEGREES = 300.0
private const val LOOP_START_TOLERANCE_DEGREES = 25.0
// Allow natural hand movement while still discouraging large camera tilt.
private const val MAX_PITCH_DRIFT_DEGREES = 14.0
private const val MAX_ROLL_DRIFT_DEGREES = 16.0
// Require a sustained tilt before rejecting a capture; ignores brief sensor jitter.
private const val LEVEL_VIOLATION_FRAMES_REQUIRED = 4


/**
 * Tracks a 360-degree orbit using the phone's rotation-vector sensor.
 * The signed accumulated yaw is used rather than a single frame delta so
 * sensor wrap-around at +/-180 degrees does not reset progress.
 */
private class OrbitCoverageTracker {
    enum class State { WAITING_FOR_DIRECTION, FORWARD, REVERSED, POSSIBLE_LOOP, COMPLETED }
    data class Snapshot(val state: State, val progressDegrees: Double, val message: String)

    private var lastYaw: Double? = null
    private var startYaw: Double? = null
    private var signedProgress = 0.0
    private var direction = 0 // +1 or -1 after direction is established
    private var oppositeProgress = 0.0
    private var reverseFrames = 0
    private var completed = false

    @Synchronized fun reset() {
        lastYaw = null
        startYaw = null
        signedProgress = 0.0
        direction = 0
        oppositeProgress = 0.0
        reverseFrames = 0
        completed = false
    }

    @Synchronized fun updateYaw(yawRadians: Double) {
        val previous = lastYaw
        lastYaw = yawRadians
        if (previous == null) {
            startYaw = yawRadians
            return
        }
        var delta = Math.toDegrees(yawRadians - previous)
        while (delta > 180.0) delta -= 360.0
        while (delta < -180.0) delta += 360.0
        if (kotlin.math.abs(delta) > 45.0) return // reject sensor jumps

        if (direction == 0 && kotlin.math.abs(delta) >= 1.0) {
            // Require a little cumulative motion before locking direction.
            signedProgress += delta
            if (kotlin.math.abs(signedProgress) >= 8.0) {
                direction = if (signedProgress >= 0.0) 1 else -1
            }
            return
        }
        if (direction == 0) return
        signedProgress += delta
        if (delta * direction < -0.5) {
            oppositeProgress += -delta * direction
            reverseFrames++
        } else {
            oppositeProgress = (oppositeProgress - kotlin.math.abs(delta)).coerceAtLeast(0.0)
            reverseFrames = 0
        }
        if (oppositeProgress >= REVERSE_WARNING_DEGREES && reverseFrames >= 4) {
            // Keep direction locked; subsequent forward movement clears this warning.
        }
        if (direction * signedProgress >= LOOP_COMPLETION_DEGREES) {
            val start = startYaw
            if (start != null) {
                var distance = Math.abs(Math.toDegrees(yawRadians - start)) % 360.0
                if (distance > 180.0) distance = 360.0 - distance
                if (distance <= LOOP_START_TOLERANCE_DEGREES) completed = true
            }
        }
    }

    @Synchronized fun snapshot(): Snapshot {
        val progress = (direction * signedProgress).coerceAtLeast(0.0).coerceAtMost(360.0)
        val state = when {
            completed -> State.COMPLETED
            progress >= LOOP_COMPLETION_DEGREES -> State.POSSIBLE_LOOP
            direction == 0 -> State.WAITING_FOR_DIRECTION
            oppositeProgress >= REVERSE_WARNING_DEGREES && reverseFrames >= 4 -> State.REVERSED
            else -> State.FORWARD
        }
        val message = when (state) {
            State.WAITING_FOR_DIRECTION -> "Start moving slowly around the object"
            State.FORWARD -> "Orbit ${progress.toInt()}° — keep moving in the same direction"
            State.REVERSED -> "Reverse movement detected — turn back and continue the orbit"
            State.POSSIBLE_LOOP -> "Almost 360° — return to the starting view to finish"
            State.COMPLETED -> "360° orbit completed"
        }
        return Snapshot(state, progress, message)
    }
}

/**
 * Keeps the phone orientation close to the orientation at scan start.
 * This checks camera tilt, NOT physical camera height. Absolute height cannot
 * be recovered reliably from a rotation sensor alone.
 */
private class CameraLevelGuard {
    data class Result(val acceptable: Boolean, val message: String)
    private var basePitch: Double? = null
    private var baseRoll: Double? = null
    private var pitchViolationFrames = 0
    private var rollViolationFrames = 0

    @Synchronized fun reset() {
        basePitch = null
        baseRoll = null
        pitchViolationFrames = 0
        rollViolationFrames = 0
    }

    @Synchronized fun update(pitchRadians: Double, rollRadians: Double) {
        if (basePitch == null || baseRoll == null) {
            basePitch = pitchRadians
            baseRoll = rollRadians
        }
    }

    @Synchronized fun check(pitchRadians: Double, rollRadians: Double): Result {
        val p0 = basePitch ?: return Result(true, "Checking camera level...")
        val r0 = baseRoll ?: return Result(true, "Checking camera level...")
        val pitchDrift = Math.toDegrees(wrapRadians(pitchRadians - p0))
        val rollDrift = Math.toDegrees(wrapRadians(rollRadians - r0))

        // Debounce the guard: brief sensor spikes should not interrupt scanning.
        pitchViolationFrames = if (kotlin.math.abs(pitchDrift) > MAX_PITCH_DRIFT_DEGREES) {
            pitchViolationFrames + 1
        } else {
            pitchViolationFrames = 0
            0
        }
        rollViolationFrames = if (kotlin.math.abs(rollDrift) > MAX_ROLL_DRIFT_DEGREES) {
            rollViolationFrames + 1
        } else {
            rollViolationFrames = 0
            0
        }

        return when {
            pitchViolationFrames >= LEVEL_VIOLATION_FRAMES_REQUIRED ->
                Result(false, "Camera tilted up/down — gently return toward the starting angle")
            rollViolationFrames >= LEVEL_VIOLATION_FRAMES_REQUIRED ->
                Result(false, "Camera tilted sideways — gently level the phone")
            else -> Result(true, "Camera angle OK")
        }
    }

    private fun wrapRadians(value: Double): Double {
        var v = value
        while (v > Math.PI) v -= 2.0 * Math.PI
        while (v < -Math.PI) v += 2.0 * Math.PI
        return v
    }
}

/** Metadata stored for each accepted still image and included in the ZIP manifest. */
private data class CapturedPhotoRecord(
    val fileName: String,
    val capturedAtMillis: Long,
    val isInitialReference: Boolean
)

private fun createScanSessionDirectory(context: Context): File {
    val appPictures = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: File(context.filesDir, "Pictures")
    val root = File(appPictures, "PhotoAssistant/Scans")
    val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val session = File(root, "Scan_$stamp")
    if (!session.mkdirs() && !session.isDirectory) {
        throw IllegalStateException("Could not create scan folder: ${session.absolutePath}")
    }
    return session
}

/**
 * Creates a ZIP containing the accepted photos and manifest.json, then publishes it
 * to Downloads/PhotoAssistant on Android 10+. Photos themselves are never added to Gallery.
 */
private fun exportScanZip(
    context: Context,
    sessionDirectory: File,
    records: List<CapturedPhotoRecord>
): String {
    if (records.isEmpty()) throw IllegalStateException("There are no accepted photos to export")

    val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val zipName = "PhotoAssistant_$stamp.zip"
    val temporaryZip = File(context.cacheDir, zipName)

    val manifest = JSONObject().apply {
        put("application", "PhotoAssistant")
        put("sessionFolder", sessionDirectory.name)
        put("createdAt", System.currentTimeMillis())
        put("photoCount", records.size)
        put("photos", JSONArray().apply {
            records.forEach { record ->
                put(JSONObject().apply {
                    put("fileName", record.fileName)
                    put("capturedAtMillis", record.capturedAtMillis)
                    put("isInitialReference", record.isInitialReference)
                })
            }
        })
        put("notes", "Images are accepted CameraX still captures. Timestamps are Unix epoch milliseconds.")
    }

    ZipOutputStream(FileOutputStream(temporaryZip)).use { zip ->
        records.forEach { record ->
            val photo = File(sessionDirectory, record.fileName)
            if (!photo.isFile) throw IllegalStateException("Missing photo: ${record.fileName}")
            zip.putNextEntry(ZipEntry(record.fileName))
            FileInputStream(photo).use { input -> input.copyTo(zip) }
            zip.closeEntry()
        }
        zip.putNextEntry(ZipEntry("manifest.json"))
        zip.write(manifest.toString(2).toByteArray(Charsets.UTF_8))
        zip.closeEntry()
    }

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, zipName)
            put(MediaStore.MediaColumns.MIME_TYPE, "application/zip")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/PhotoAssistant")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("Could not create ZIP in Downloads")
        try {
            context.contentResolver.openOutputStream(uri)?.use { output ->
                FileInputStream(temporaryZip).use { input -> input.copyTo(output) }
            } ?: throw IllegalStateException("Could not write ZIP to Downloads")
            val completed = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
            context.contentResolver.update(uri, completed, null, null)
        } catch (e: Exception) {
            context.contentResolver.delete(uri, null, null)
            throw e
        }
        temporaryZip.delete()
        return "Downloads/PhotoAssistant/$zipName"
    } else {
        // Legacy fallback: save in app-specific external files, where no storage permission is needed.
        val outputDir = File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "PhotoAssistant")
        if (!outputDir.exists() && !outputDir.mkdirs()) throw IllegalStateException("Could not create export folder")
        val destination = File(outputDir, zipName)
        temporaryZip.copyTo(destination, overwrite = true)
        temporaryZip.delete()
        return destination.absolutePath
    }
}

private data class OrbFeatures(
    val keypoints: MatOfKeyPoint,
    val descriptors: Mat
) {
    fun release() {
        keypoints.release()
        descriptors.release()
    }
}

private data class MatchMetrics(
    val goodMatches: Int,
    val inliers: Int,
    val inlierRatio: Double,
    val averageMovement: Double
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                CameraScreen()
            }
        }
    }
}

private data class ScanProjectItem(
    val folder: File,
    val photoCount: Int,
    val modifiedAt: Long
)

private fun loadScanProjects(context: Context): List<ScanProjectItem> {
    val appPictures = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        ?: File(context.filesDir, "Pictures")
    val root = File(appPictures, "PhotoAssistant/Scans")
    return root.listFiles()
        ?.filter { it.isDirectory }
        ?.map { folder ->
            ScanProjectItem(
                folder = folder,
                photoCount = folder.listFiles()?.count {
                    it.isFile && it.extension.equals("jpg", ignoreCase = true)
                } ?: 0,
                modifiedAt = folder.lastModified()
            )
        }
        ?.sortedByDescending { it.modifiedAt }
        ?: emptyList()
}

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    var showCamera by remember { mutableStateOf(false) }
    var permissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    var projects by remember { mutableStateOf(loadScanProjects(context)) }
    var projectPendingDelete by remember { mutableStateOf<ScanProjectItem?>(null) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        permissionGranted = granted
        if (granted) showCamera = true
    }

    LaunchedEffect(showCamera) {
        if (showCamera && !permissionGranted) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (showCamera && permissionGranted) {
        CameraPreview(
            modifier = Modifier.fillMaxSize(),
            onCancel = {
                showCamera = false
                projects = loadScanProjects(context)
            }
        )
        return
    }

    projectPendingDelete?.let { project ->
        AlertDialog(
            onDismissRequest = { projectPendingDelete = null },
            title = { Text("Delete project?") },
            text = {
                Text("Delete ${project.folder.name} and its stored photos from this device? Any ZIP already exported to Downloads will remain.")
            },
            confirmButton = {
                TextButton(onClick = {
                    val deleted = project.folder.deleteRecursively()
                    projects = loadScanProjects(context)
                    projectPendingDelete = null
                    if (!deleted) {
                        android.widget.Toast.makeText(
                            context,
                            "Could not delete project folder",
                            android.widget.Toast.LENGTH_SHORT
                        ).show()
                    }
                }) { Text("Delete", color = Color(0xFFD9534F)) }
            },
            dismissButton = {
                TextButton(onClick = { projectPendingDelete = null }) { Text("Cancel") }
            },
            containerColor = Color(0xFF303030),
            titleContentColor = Color.White,
            textContentColor = Color(0xFFDDDDDD)
        )
    }

    val charcoal = Color(0xFF242424)
    val cardColor = Color(0xFF303030)
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = charcoal
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(start = 22.dp, end = 22.dp, top = 34.dp, bottom = 116.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp)
            ) {
                Text(
                    text = "PhotoAssistant",
                    color = Color.White,
                    style = MaterialTheme.typography.headlineMedium
                )
                Text(
                    text = "Photogrammetry capture projects",
                    color = Color(0xFFBDBDBD),
                    style = MaterialTheme.typography.bodyMedium
                )
                Text(
                    text = "Previous projects",
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium
                )

                if (projects.isEmpty()) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(18.dp))
                            .background(cardColor)
                            .padding(22.dp),
                        contentAlignment = Alignment.CenterStart
                    ) {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("▧", color = Color.White, fontSize = 32.sp)
                            Text("No projects yet", color = Color.White, style = MaterialTheme.typography.titleMedium)
                            Text(
                                "Your completed scan sessions will appear here.",
                                color = Color(0xFFBDBDBD),
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(projects, key = { it.folder.absolutePath }) { project ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(16.dp))
                                    .background(cardColor)
                                    .padding(14.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(14.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(72.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Color(0xFF424242)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text("▧", color = Color.White, fontSize = 34.sp)
                                }
                                Column(
                                    modifier = Modifier.weight(1f),
                                    verticalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(project.folder.name, color = Color.White, style = MaterialTheme.typography.titleSmall)
                                    Text("${project.photoCount} photos", color = Color(0xFFBDBDBD), style = MaterialTheme.typography.bodyMedium)
                                    Text(
                                        SimpleDateFormat("dd MMM yyyy · HH:mm", Locale.getDefault()).format(Date(project.modifiedAt)),
                                        color = Color(0xFF9E9E9E),
                                        style = MaterialTheme.typography.bodySmall
                                    )
                                }
                                Box(
                                    modifier = Modifier
                                        .size(44.dp)
                                        .clip(CircleShape)
                                        .background(Color(0xFF414141))
                                        .clickable { projectPendingDelete = project },
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text("🗑", fontSize = 20.sp)
                                }
                            }
                        }
                    }
                }
            }

            Button(
                onClick = {
                    if (permissionGranted) {
                        showCamera = true
                    } else {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .padding(start = 24.dp, end = 24.dp, bottom = 32.dp)
                    .height(58.dp),
                shape = RoundedCornerShape(18.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = Color.Black
                )
            ) {
                Text("＋   Start new scan", style = MaterialTheme.typography.titleMedium)
            }
        }
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    onCancel: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember(context) {
        ContextCompat.getMainExecutor(context)
    }

    val analysisExecutor = remember {
        Executors.newSingleThreadExecutor()
    }

    val imageCaptureRef =
        remember { AtomicReference<ImageCapture?>(null) }

    // The accepted reference is shared with the analysis thread.
    val referenceFeatures =
        remember { AtomicReference<OrbFeatures?>(null) }
    // Keeps accepted viewpoints so returning to an already captured angle
    // does not create another photo.
    val rememberedViews = remember { mutableListOf<OrbFeatures>() }
    val orbitTracker = remember { OrbitCoverageTracker() }
    val levelGuard = remember { CameraLevelGuard() }
    val currentPitch = remember { AtomicReference<Double?>(null) }
    val currentRoll = remember { AtomicReference<Double?>(null) }
    val sensorManager = remember(context) {
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    }
    val rotationSensor = remember(sensorManager) {
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    }

    val scanning = remember { AtomicBoolean(false) }
    val captureInProgress = remember { AtomicBoolean(false) }
    val badFrameCount = remember { AtomicInteger(0) }
    val goodFrameCount = remember { AtomicInteger(0) }

    val lastCaptureTime = remember { AtomicLong(0L) }
    val lastUiUpdateTime = remember { AtomicLong(0L) }
    val sessionDirectoryRef = remember { AtomicReference<File?>(null) }
    val capturedPhotoRecords = remember { mutableListOf<CapturedPhotoRecord>() }
    val exportExecutor = remember { Executors.newSingleThreadExecutor() }

    var isScanning by remember { mutableStateOf(false) }
    var statusText by remember {
        mutableStateOf("Ready — press Start Scan")
    }
    var statusColor by remember {
        mutableStateOf(Color(0xFFFFC107))
    }
    var acceptedPhotoCount by remember {
        mutableIntStateOf(0)
    }
    var isExporting by remember { mutableStateOf(false) }

    fun updateStatus(
        message: String,
        color: Color,
        force: Boolean = false
    ) {
        val now = System.currentTimeMillis()
        val previous = lastUiUpdateTime.get()

        if (!force && now - previous < 250L) return

        lastUiUpdateTime.set(now)

        mainExecutor.execute {
            statusText = message
            statusColor = color
        }
    }

    /*
     * Captures a photo only when requested.
     * The first capture initializes the reference.
     * Later captures are triggered by the live analyzer.
     */
    val capturePhoto: (Boolean, OrbFeatures?) -> Unit = capturePhoto@ { isInitial, acceptedFeatures ->
        val imageCapture = imageCaptureRef.get()

        if (imageCapture == null) {
            acceptedFeatures?.release()
            updateStatus(
                "Camera is not ready",
                Color(0xFFFFC107),
                true
            )
        } else if (
            captureInProgress.compareAndSet(false, true)
        ) {
            updateStatus(
                if (isInitial) {
                    "Capturing initial reference..."
                } else {
                    "Capturing accepted frame..."
                },
                Color(0xFFFFC107),
                true
            )

            val sessionDirectory = sessionDirectoryRef.get()
            if (sessionDirectory == null) {
                acceptedFeatures?.release()
                captureInProgress.set(false)
                updateStatus("Scan folder is not ready — restart the scan", Color.Red, true)
                return@capturePhoto
            }
            val photoNumber = synchronized(capturedPhotoRecords) { capturedPhotoRecords.size + 1 }
            val photoFile = File(sessionDirectory, "%04d.jpg".format(Locale.US, photoNumber))
            val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

            imageCapture.takePicture(
                outputOptions,
                mainExecutor,
                object : ImageCapture.OnImageSavedCallback {

                    override fun onImageSaved(
                        output: ImageCapture.OutputFileResults
                    ) {
                        val savedFile = photoFile

                        if (!savedFile.isFile || savedFile.length() == 0L) {
                            acceptedFeatures?.release()
                            captureInProgress.set(false)
                            updateStatus(
                                "Could not access saved photo",
                                Color.Red,
                                true
                            )
                            return
                        }

                        if (!scanning.get()) {
                            savedFile.delete()
                            acceptedFeatures?.release()
                            captureInProgress.set(false)
                            return
                        }

                        val capturedAt = System.currentTimeMillis()
                        synchronized(capturedPhotoRecords) {
                            capturedPhotoRecords.add(
                                CapturedPhotoRecord(
                                    fileName = savedFile.name,
                                    capturedAtMillis = capturedAt,
                                    isInitialReference = isInitial
                                )
                            )
                        }
                        Log.i(TAG, "Photo saved in app session: ${savedFile.absolutePath}")

                        // Keep ORB features from the live analysis frame as the
                        // reference. Extracting features from the saved still image
                        // mixes ImageCapture and ImageAnalysis orientations/resolutions.
                        analysisExecutor.execute {
                            if (isInitial) {
                                // The baseline was initialized from the live frame
                                // before the initial still photo was requested.
                                lastCaptureTime.set(System.currentTimeMillis())
                                badFrameCount.set(0)
                                goodFrameCount.set(0)
                                captureInProgress.set(false)

                                mainExecutor.execute {
                                    acceptedPhotoCount += 1
                                    statusText = "Reference ready — move slowly"
                                    statusColor = Color(0xFF4CAF50)
                                }
                                Log.i(TAG, "Initial reference retained from live analysis frame")
                            } else if (acceptedFeatures == null) {
                                savedFile.delete()
                                captureInProgress.set(false)
                                updateStatus("Could not update tracking reference", Color.Red, true)
                            } else {
                                // Keep the accepted features in the viewpoint history.
                                // Do not release the old reference: it remains useful for
                                // detecting a return to a previously captured angle.
                                // Keep every accepted view for the duration of this scan.
                                // This allows loop-closure checks against the very first
                                // photo as well as every later accepted photo.
                                synchronized(rememberedViews) {
                                    rememberedViews.add(acceptedFeatures)
                                }
                                referenceFeatures.set(acceptedFeatures)

                                lastCaptureTime.set(System.currentTimeMillis())
                                badFrameCount.set(0)
                                goodFrameCount.set(0)
                                captureInProgress.set(false)

                                mainExecutor.execute {
                                    acceptedPhotoCount += 1
                                    statusText = "Photo accepted — keep moving"
                                    statusColor = Color(0xFF4CAF50)
                                }
                                Log.i(TAG, "Reference updated from accepted live frame: ${acceptedFeatures.descriptors.rows()} descriptors")
                            }
                        }
                    }

                    override fun onError(
                        exception: ImageCaptureException
                    ) {
                        acceptedFeatures?.release()
                        captureInProgress.set(false)

                        Log.e(TAG, "Photo capture failed", exception)

                        updateStatus(
                            "Capture failed — try again",
                            Color.Red,
                            true
                        )
                    }
                }
            )
        } else {
            // The frame was not used for a capture; release its copied Mats.
            acceptedFeatures?.release()
        }
    }

    DisposableEffect(sensorManager, rotationSensor) {
        val listener = object : SensorEventListener {
            private val rotationMatrix = FloatArray(9)
            private val orientation = FloatArray(3)
            override fun onSensorChanged(event: SensorEvent) {
                if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR && scanning.get()) {
                    SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                    SensorManager.getOrientation(rotationMatrix, orientation)
                    orbitTracker.updateYaw(orientation[0].toDouble())
                    currentPitch.set(orientation[1].toDouble())
                    currentRoll.set(orientation[2].toDouble())
                    levelGuard.update(orientation[1].toDouble(), orientation[2].toDouble())
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        if (rotationSensor != null) {
            sensorManager.registerListener(listener, rotationSensor, SensorManager.SENSOR_DELAY_GAME)
        }
        onDispose { sensorManager.unregisterListener(listener) }
    }

    DisposableEffect(analysisExecutor) {
        onDispose {
            scanning.set(false)
            imageCaptureRef.set(null)
            runCatching {
                val providerFuture = ProcessCameraProvider.getInstance(context)
                providerFuture.addListener(
                    { runCatching { providerFuture.get().unbindAll() } },
                    mainExecutor
                )
            }

            // Every reference is also stored in the history, so release each
            // remembered feature set exactly once.
            synchronized(rememberedViews) {
                rememberedViews.forEach { it.release() }
                rememberedViews.clear()
            }
            referenceFeatures.set(null)

            analysisExecutor.shutdown()
            exportExecutor.shutdown()
        }
    }

    Box(modifier = modifier) {

        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { viewContext ->
                val previewView = PreviewView(viewContext).apply {
                    scaleType =
                        PreviewView.ScaleType.FILL_CENTER
                }

                val providerFuture =
                    ProcessCameraProvider.getInstance(viewContext)

                providerFuture.addListener(
                    {
                        try {
                            val cameraProvider = providerFuture.get()

                            val preview = Preview.Builder()
                                .build()
                                .also {
                                    it.surfaceProvider =
                                        previewView.surfaceProvider
                                }

                            val imageCapture =
                                ImageCapture.Builder()
                                    .setCaptureMode(
                                        ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
                                    )
                                    .build()

                            imageCaptureRef.set(imageCapture)

                            val imageAnalysis =
                                ImageAnalysis.Builder()
                                    .setBackpressureStrategy(
                                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                                    )
                                    .build()

                            imageAnalysis.setAnalyzer(
                                analysisExecutor
                            ) { image ->

                                try {
                                    if (!scanning.get()) {
                                        return@setAnalyzer
                                    }

                                    if (captureInProgress.get()) {
                                        return@setAnalyzer
                                    }

                                    val sharpness = calculateLaplacianVariance(image)
                                    val current = extractOrbFromImageProxy(image)

                                    if (current == null) {
                                        updateStatus("Cannot analyze current frame", Color(0xFFFFC107))
                                        return@setAnalyzer
                                    }

                                    try {
                                        val reference = referenceFeatures.get()

                                        // Initialize the baseline from the same ImageAnalysis
                                        // stream used for every subsequent comparison. This avoids
                                        // comparing a rotated high-resolution still to raw Y-plane frames.
                                        if (reference == null) {
                                            val baseline = copyOrbFeatures(current)
                                            if (referenceFeatures.compareAndSet(null, baseline)) {
                                                synchronized(rememberedViews) {
                                                    rememberedViews.add(baseline)
                                                }
                                                badFrameCount.set(0)
                                                goodFrameCount.set(0)
                                                updateStatus("Reference found — saving first photo...", Color(0xFFFFC107), true)
                                                mainExecutor.execute {
                                                    if (scanning.get()) capturePhoto(true, null)
                                                }
                                            } else {
                                                baseline.release()
                                            }
                                            return@setAnalyzer
                                        }

                                        if (sharpness == null) {
                                            updateStatus("Checking image quality...", Color(0xFFFFC107))
                                            return@setAnalyzer
                                        }

                                        val metrics =
                                            matchOrbFeatures(
                                                reference,
                                                current
                                            )

                                        if (metrics == null) {
                                            val failures =
                                                badFrameCount.incrementAndGet()
                                            goodFrameCount.set(0)

                                            val revisitingOldView = isPreviouslyCapturedView(
                                                current, rememberedViews
                                            )
                                            if (revisitingOldView) {
                                                updateStatus(
                                                    "Previously captured view detected — move back toward the last accepted view",
                                                    Color(0xFFFFC107),
                                                    true
                                                )
                                            } else if (failures >= 3) {
                                                updateStatus(
                                                    "Tracking lost — return to the previous view",
                                                    Color.Red
                                                )
                                            } else {
                                                updateStatus(
                                                    "Checking tracking...",
                                                    Color(0xFFFFC107)
                                                )
                                            }

                                            return@setAnalyzer
                                        }

                                        val qualityOk =
                                            sharpness >= MIN_SHARPNESS

                                        val trackingOk =
                                            metrics.goodMatches >=
                                                    MIN_GOOD_MATCHES &&
                                                    metrics.inliers >= MIN_INLIERS &&
                                                    metrics.inlierRatio >=
                                                    MIN_INLIER_RATIO

                                        if (!trackingOk) {
                                            val failures =
                                                badFrameCount.incrementAndGet()
                                            goodFrameCount.set(0)

                                            val revisitingOldView = isPreviouslyCapturedView(
                                                current, rememberedViews
                                            )
                                            if (revisitingOldView) {
                                                updateStatus(
                                                    "Previously captured view detected — move back toward the last accepted view",
                                                    Color(0xFFFFC107),
                                                    true
                                                )
                                            } else if (failures >= 3) {
                                                updateStatus(
                                                    "Poor overlap — move back toward the previous view",
                                                    Color.Red
                                                )
                                            } else {
                                                updateStatus(
                                                    "Tracking uncertain — move slowly",
                                                    Color(0xFFFFC107)
                                                )
                                            }

                                            return@setAnalyzer
                                        }

                                        badFrameCount.set(0)
                                        goodFrameCount.incrementAndGet()

                                        if (!qualityOk) {
                                            updateStatus(
                                                "Image blurry — hold the camera steady",
                                                Color(0xFFFFC107)
                                            )
                                            return@setAnalyzer
                                        }

                                        // Reject captures if the phone has tilted too far from
                                        // its starting orientation. This is an orientation guard,
                                        // not a measurement of physical height.
                                        if (rotationSensor != null) {
                                            val pitch = currentPitch.get()
                                            val roll = currentRoll.get()
                                            if (pitch != null && roll != null) {
                                                val level = levelGuard.check(pitch, roll)
                                                if (!level.acceptable) {
                                                    goodFrameCount.set(0)
                                                    updateStatus(level.message, Color(0xFFFFC107), true)
                                                    return@setAnalyzer
                                                }
                                            }
                                        }

                                        val orbit = orbitTracker.snapshot()
                                        if (orbit.state == OrbitCoverageTracker.State.REVERSED) {
                                            goodFrameCount.set(0)
                                            updateStatus(orbit.message, Color(0xFFFFC107), true)
                                            return@setAnalyzer
                                        }
                                        if (orbit.state == OrbitCoverageTracker.State.COMPLETED) {
                                            scanning.set(false)
                                            updateStatus(orbit.message, Color(0xFF4CAF50), true)
                                            mainExecutor.execute { isScanning = false }
                                            return@setAnalyzer
                                        }
                                        if (orbit.state == OrbitCoverageTracker.State.POSSIBLE_LOOP) {
                                            updateStatus(orbit.message, Color(0xFFFFC107), true)
                                        } else {
                                            updateStatus(orbit.message, Color(0xFF4CAF50))
                                        }

                                        val now =
                                            System.currentTimeMillis()

                                        val cooldownPassed =
                                            now - lastCaptureTime.get() >=
                                                    CAPTURE_COOLDOWN_MS

                                        val enoughMovement =
                                            metrics.averageMovement >=
                                                    MIN_MOVEMENT_PIXELS

                                        val stableTracking =
                                            goodFrameCount.get() >= 5

                                        if (
                                            stableTracking &&
                                            (rotationSensor == null || orbit.state == OrbitCoverageTracker.State.FORWARD) &&
                                            cooldownPassed &&
                                            enoughMovement &&
                                            scanning.get() &&
                                            !captureInProgress.get()
                                        ) {
                                            // Only check the viewpoint history when a new
                                            // capture is otherwise about to happen. This avoids
                                            // running extra matching for every camera frame.
                                            val returnedToPreviousView = isPreviouslyCapturedView(
                                                current, rememberedViews
                                            )

                                            if (returnedToPreviousView) {
                                                goodFrameCount.set(0)
                                                updateStatus(
                                                    "Already captured this view — return to the last view and continue slowly",
                                                    Color(0xFFFFC107),
                                                    true
                                                )
                                                return@setAnalyzer
                                            }

                                            // Copy now, while the current frame's Mats are valid.
                                            val acceptedSnapshot = copyOrbFeatures(current)
                                            mainExecutor.execute {
                                                if (scanning.get()) {
                                                    capturePhoto(false, acceptedSnapshot)
                                                } else {
                                                    acceptedSnapshot.release()
                                                }
                                            }
                                        }

                                        Log.d(
                                            TAG,
                                            "TRACKING sharpness=${"%.1f".format(Locale.US, sharpness)}, " +
                                                    "matches=${metrics.goodMatches}, " +
                                                    "inliers=${metrics.inliers}, " +
                                                    "inlierRatio=${"%.2f".format(Locale.US, metrics.inlierRatio)}, " +
                                                    "movement=${"%.1f".format(Locale.US, metrics.averageMovement)}, " +
                                                    "captureThreshold=$MIN_MOVEMENT_PIXELS, " +
                                                    "stableFrames=${goodFrameCount.get()}"
                                        )
                                    } finally {
                                        // Current-frame features are temporary.
                                        current.release()
                                    }
                                } catch (e: Exception) {
                                    Log.e(
                                        TAG,
                                        "Live analysis failed",
                                        e
                                    )
                                    updateStatus(
                                        "Analysis error — check Logcat",
                                        Color.Red
                                    )
                                } finally {
                                    image.close()
                                }
                            }

                            cameraProvider.unbindAll()

                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageCapture,
                                imageAnalysis
                            )

                            Log.i(TAG, "Camera initialized")
                        } catch (e: Exception) {
                            Log.e(TAG, "Camera setup failed", e)

                            updateStatus(
                                "Camera setup failed",
                                Color.Red,
                                true
                            )
                        }
                    },
                    ContextCompat.getMainExecutor(viewContext)
                )

                previewView
            }
        )

        // Cancel the current scan and return to the lobby. This abandons the
        // current session instead of exporting a partial ZIP.
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 142.dp, end = 18.dp)
                .size(44.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = 0.42f))
                .clickable {
                    scanning.set(false)
                    isScanning = false
                    val session = sessionDirectoryRef.getAndSet(null)
                    session?.deleteRecursively()
                    synchronized(capturedPhotoRecords) { capturedPhotoRecords.clear() }
                    acceptedPhotoCount = 0
                    onCancel()
                },
            contentAlignment = Alignment.Center
        ) {
            Text("×", color = Color.Black, fontSize = 32.sp, lineHeight = 32.sp)
        }

        // Status overlay
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(16.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(Color.Black.copy(alpha = 0.78f))
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = statusText,
                color = statusColor,
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = "Accepted photos: $acceptedPhotoCount",
                color = Color.White,
                style = MaterialTheme.typography.bodyMedium
            )
        }

        // Start / Stop controls
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = 64.dp),
            horizontalArrangement = Arrangement.spacedBy(
                16.dp,
                Alignment.CenterHorizontally
            ),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(
                enabled = !isScanning && !isExporting,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.Black,
                    contentColor = Color.White,
                    disabledContainerColor = Color(0xFF333333),
                    disabledContentColor = Color.LightGray
                ),
                onClick = {
                    try {
                        val session = createScanSessionDirectory(context)
                        sessionDirectoryRef.set(session)
                        synchronized(capturedPhotoRecords) { capturedPhotoRecords.clear() }
                        acceptedPhotoCount = 0
                    } catch (e: Exception) {
                        Log.e(TAG, "Could not create scan session", e)
                        updateStatus("Could not create scan folder: ${e.message}", Color.Red, true)
                        return@Button
                    }
                    scanning.set(true)
                    isScanning = true

                    synchronized(rememberedViews) {
                        rememberedViews.forEach { it.release() }
                        rememberedViews.clear()
                    }
                    // The active reference is also in rememberedViews, so do not
                    // release it separately (that would release the same Mats twice).
                    referenceFeatures.set(null)

                    badFrameCount.set(0)
                    goodFrameCount.set(0)
                    lastCaptureTime.set(0L)
                    orbitTracker.reset()
                    levelGuard.reset()
                    currentPitch.set(null)
                    currentRoll.set(null)

                    updateStatus(
                        if (rotationSensor == null) "Starting scan — rotation sensor unavailable; direction guidance disabled" else "Starting scan — finding a clear reference frame...",
                        Color(0xFFFFC107),
                        true
                    )
                }
            ) {
                Text("Start Scan")
            }

            Button(
                enabled = !isExporting && acceptedPhotoCount > 0 && !captureInProgress.get(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.Black,
                    contentColor = Color.White,
                    disabledContainerColor = Color(0xFF333333),
                    disabledContentColor = Color.LightGray
                ),
                onClick = {
                    scanning.set(false)
                    isScanning = false
                    isExporting = true
                    updateStatus("Preparing ZIP for computer transfer...", Color(0xFFFFC107), true)
                    val session = sessionDirectoryRef.get()
                    val records = synchronized(capturedPhotoRecords) { capturedPhotoRecords.toList() }
                    if (session == null || records.isEmpty()) {
                        isExporting = false
                        updateStatus("No photos to export", Color.Red, true)
                    } else {
                        exportExecutor.execute {
                            try {
                                val location = exportScanZip(context, session, records)
                                mainExecutor.execute {
                                    isExporting = false
                                    updateStatus("ZIP ready: $location", Color(0xFF4CAF50), true)
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "ZIP export failed", e)
                                mainExecutor.execute {
                                    isExporting = false
                                    updateStatus("ZIP export failed: ${e.message}", Color.Red, true)
                                }
                            }
                        }
                    }
                }
            ) {
                Text(if (isExporting) "Creating ZIP..." else "Finish & Export ZIP")
            }
        }
    }
}

private fun isPreviouslyCapturedView(
    current: OrbFeatures,
    rememberedViews: List<OrbFeatures>
): Boolean {
    synchronized(rememberedViews) {
        // Compare against ALL remembered views, including the active reference
        // and the very first view. Never skip a view just because it is current.
        for (oldView in rememberedViews) {
            val metrics = matchOrbFeatures(oldView, current) ?: continue
            if (
                metrics.goodMatches >= MIN_GOOD_MATCHES &&
                metrics.inliers >= MIN_INLIERS &&
                metrics.inlierRatio >= MIN_INLIER_RATIO &&
                metrics.averageMovement <= DUPLICATE_VIEW_MOVEMENT_PIXELS
            ) {
                Log.i(TAG, "Previously captured viewpoint recognized")
                return true
            }
        }
    }
    return false
}

private fun copyOrbFeatures(source: OrbFeatures): OrbFeatures {
    val copiedKeypoints = MatOfKeyPoint()
    copiedKeypoints.fromArray(*source.keypoints.toArray())
    return OrbFeatures(copiedKeypoints, source.descriptors.clone())
}

/*
 * Extract ORB features directly from a live CameraX YUV frame.
 * The Y plane already contains grayscale intensity values.
 */
private fun extractOrbFromImageProxy(
    image: ImageProxy
): OrbFeatures? {
    if (!OpenCVLoader.initLocal()) {
        Log.e(TAG, "OpenCV initialization failed")
        return null
    }

    if (image.format != ImageFormat.YUV_420_888) {
        Log.w(TAG, "Unsupported frame format: ${image.format}")
        return null
    }

    val width = image.width
    val height = image.height
    val plane = image.planes[0]
    val buffer = plane.buffer.duplicate()
    val rowStride = plane.rowStride
    val startPosition = buffer.position()

    val pixels = ByteArray(width * height)

    for (y in 0 until height) {
        val sourceOffset = startPosition + y * rowStride
        val targetOffset = y * width

        for (x in 0 until width) {
            pixels[targetOffset + x] =
                buffer.get(sourceOffset + x)
        }
    }

    val gray = Mat(height, width, org.opencv.core.CvType.CV_8UC1)
    val keypoints = MatOfKeyPoint()
    val descriptors = Mat()
    val mask = Mat()
    val orb = ORB.create(ORB_FEATURE_COUNT)

    return try {
        gray.put(0, 0, pixels)

        orb.detectAndCompute(
            gray,
            mask,
            keypoints,
            descriptors
        )

        if (descriptors.empty() || keypoints.empty()) {
            keypoints.release()
            descriptors.release()
            null
        } else {
            OrbFeatures(keypoints, descriptors)
        }
    } catch (e: Exception) {
        Log.e(TAG, "Live ORB extraction failed", e)
        keypoints.release()
        descriptors.release()
        null
    } finally {
        gray.release()
        mask.release()
    }
}

/*
 * KNN matching + Ratio Test + Fundamental Matrix RANSAC.
 * Returns null if there are too few matches or estimation fails.
 */
private fun matchOrbFeatures(
    reference: OrbFeatures,
    current: OrbFeatures
): MatchMetrics? {
    val referenceDescriptors = reference.descriptors
    val currentDescriptors = current.descriptors

    if (
        referenceDescriptors.empty() ||
        currentDescriptors.empty()
    ) {
        return null
    }

    val matcher = BFMatcher.create(Core.NORM_HAMMING, false)
    val knnMatches = ArrayList<MatOfDMatch>()

    val points1 = MatOfPoint2f()
    val points2 = MatOfPoint2f()
    val inlierMask = Mat()

    var fundamentalMatrix: Mat? = null

    try {
        matcher.knnMatch(
            referenceDescriptors,
            currentDescriptors,
            knnMatches,
            2
        )

        val goodMatches = knnMatches.mapNotNull { group ->
            val pair = group.toArray()

            if (
                pair.size >= 2 &&
                pair[0].distance <
                RATIO_THRESHOLD * pair[1].distance
            ) {
                pair[0]
            } else {
                null
            }
        }

        if (goodMatches.size < 8) {
            return null
        }

        val referenceKeypoints = reference.keypoints.toArray()
        val currentKeypoints = current.keypoints.toArray()

        val matchedPoints1 = ArrayList<Point>()
        val matchedPoints2 = ArrayList<Point>()

        for (match in goodMatches) {
            if (
                match.queryIdx !in referenceKeypoints.indices ||
                match.trainIdx !in currentKeypoints.indices
            ) {
                continue
            }

            matchedPoints1.add(
                referenceKeypoints[match.queryIdx].pt
            )
            matchedPoints2.add(
                currentKeypoints[match.trainIdx].pt
            )
        }

        if (matchedPoints1.size < 8) {
            return null
        }

        points1.fromList(matchedPoints1)
        points2.fromList(matchedPoints2)

        fundamentalMatrix = Calib3d.findFundamentalMat(
            points1,
            points2,
            Calib3d.FM_RANSAC,
            3.0,
            0.99,
            inlierMask
        )

        val matrix = fundamentalMatrix

        if (
            matrix == null ||
            matrix.empty() ||
            inlierMask.empty()
        ) {
            return null
        }

        val inliers = Core.countNonZero(inlierMask)
        val total = matchedPoints1.size

        if (inliers == 0 || total == 0) {
            return null
        }

        // Estimate image displacement using RANSAC inliers only.
        var displacementSum = 0.0
        var displacementCount = 0

        for (i in matchedPoints1.indices) {
            val maskValue = if (inlierMask.rows() == 1) {
                inlierMask.get(0, i)?.get(0) ?: 0.0
            } else {
                inlierMask.get(i, 0)?.get(0) ?: 0.0
            }

            if (maskValue != 0.0) {
                val p1 = matchedPoints1[i]
                val p2 = matchedPoints2[i]

                val dx = p2.x - p1.x
                val dy = p2.y - p1.y

                displacementSum += kotlin.math.sqrt(
                    dx * dx + dy * dy
                )

                displacementCount++
            }
        }

        val averageMovement =
            if (displacementCount > 0) {
                displacementSum / displacementCount
            } else {
                0.0
            }

        val result = MatchMetrics(
            goodMatches = total,
            inliers = inliers,
            inlierRatio = inliers.toDouble() / total,
            averageMovement = averageMovement
        )

        Log.d(
            TAG,
            "Matches=${result.goodMatches}, " +
                    "inliers=${result.inliers}, " +
                    "ratio=${result.inlierRatio}, " +
                    "movement=${result.averageMovement}"
        )

        return result
    } catch (e: Exception) {
        Log.e(TAG, "Feature matching/RANSAC failed", e)
        return null
    } finally {
        knnMatches.forEach { it.release() }

        points1.release()
        points2.release()
        inlierMask.release()

        fundamentalMatrix?.release()
        matcher.clear()
    }
}

/*
 * Approximate sharpness score computed from the Y plane.
 * The threshold depends on camera resolution and must be calibrated.
 */
private fun calculateLaplacianVariance(
    image: ImageProxy
): Double? {
    if (image.format != ImageFormat.YUV_420_888) {
        return null
    }

    val width = image.width
    val height = image.height
    val plane = image.planes[0]
    val buffer = plane.buffer.duplicate()
    val rowStride = plane.rowStride
    val startPosition = buffer.position()

    if (width < 3 || height < 3) return null

    val step = maxOf(1, maxOf(width, height) / 320)

    fun pixel(x: Int, y: Int): Double {
        val index = startPosition + y * rowStride + x
        return (buffer.get(index).toInt() and 0xFF).toDouble()
    }

    var sum = 0.0
    var sumSquared = 0.0
    var count = 0L

    var y = step

    while (y < height - step) {
        var x = step

        while (x < width - step) {
            val laplacian =
                pixel(x - step, y) +
                        pixel(x + step, y) +
                        pixel(x, y - step) +
                        pixel(x, y + step) -
                        4.0 * pixel(x, y)

            sum += laplacian
            sumSquared += laplacian * laplacian
            count++

            x += step
        }

        y += step
    }

    if (count == 0L) return null

    val mean = sum / count
    val variance = (sumSquared / count) - (mean * mean)

    return variance.coerceAtLeast(0.0)
}
