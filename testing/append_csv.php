<?php
// append_csv.php — PHP 5.3+ compatible (no [] arrays, no ?? operator)

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(204); exit; }

$csv_file = dirname(__FILE__) . '/testresults.csv';

// --------- DEBUG AID ---------
// Visit: append_csv.php?debug=1 to see what PHP sees
if (isset($_GET['debug'])) {
  $raw = file_get_contents('php://input');
  $info = array(
    'method'       => isset($_SERVER['REQUEST_METHOD']) ? $_SERVER['REQUEST_METHOD'] : '',
    'content_type' => isset($_SERVER['CONTENT_TYPE']) ? $_SERVER['CONTENT_TYPE']
                       : (isset($_SERVER['HTTP_CONTENT_TYPE']) ? $_SERVER['HTTP_CONTENT_TYPE'] : ''),
    'get_keys'     => array_keys($_GET),
    'post_keys'    => array_keys($_POST),
    'raw_len'      => strlen($raw),
    'raw_sample'   => substr($raw, 0, 200),
  );
  echo json_encode($info);
  exit;
}

// --------- Acquire "line" from multiple possible places ---------
$line = null;

// 1) Accept GET ?line=...
if (isset($_GET['line'])) {
  $line = $_GET['line'];
}

// 2) Accept POST form-encoded
if ($line === null && isset($_POST['line'])) {
  $line = $_POST['line'];
}

// 3) Accept POST JSON: {"line":"..."} or {"data":{"line":"..."}}
$raw = null;
if ($line === null) {
  $raw = file_get_contents('php://input');
  if ($raw !== false && $raw !== '') {
    $json = json_decode($raw, true);
    if (function_exists('json_last_error') && json_last_error() === JSON_ERROR_NONE && is_array($json)) {
      if (isset($json['line'])) {
        $line = $json['line'];
      } else if (isset($json['data']) && is_array($json['data']) && isset($json['data']['line'])) {
        $line = $json['data']['line'];
      }
    }
  }
}

// 4) Some older PHP/hosts don’t populate $_POST for form-encoded bodies (php://input still has it)
// Try to parse raw form body ourselves
if ($line === null) {
  if ($raw === null) { $raw = file_get_contents('php://input'); }
  if ($raw !== false && $raw !== '') {
    $parsed = array();
    parse_str($raw, $parsed);
    if (isset($parsed['line'])) {
      $line = $parsed['line'];
    }
  }
}

// 5) Last resort: treat the entire raw body as the line
if ($line === null) {
  if ($raw === null) { $raw = file_get_contents('php://input'); }
  if ($raw !== false && $raw !== '') {
    $line = $raw;
  }
}

// --------- Clean & validate ---------
$line = (string)$line;
$line = preg_replace("/[\r\n]+/", " ", $line);
if ($line === '' || strlen($line) === 0) {
  http_response_code(400);
  echo json_encode(array('ok' => false, 'error' => 'empty'));
  exit;
}

// --------- Append ---------
// Ensure file is writable by the web server user (permissions/ownership).
// Optionally write a header row if the file does not exist yet:
// if (!file_exists($csv_file)) {
//   $header = "Test_number|Model|Question_text|Response_text|General_comment|Response_comment|Response_accuracy|Response_relevance|Response_style|Reference1_relevance|Reference1_comment|Reference2_relevance|Reference2_comment|Reference3_relevance|Reference3_comment|Reference4_relevance|Reference4_comment|Reference5_relevance|Reference5_comment|Reference6_relevance|Reference6_comment|Reference7_relevance|Reference7_comment|Reference8_relevance|Reference8_comment|Reference9_relevance|Reference9_comment|Reference10_relevance|Reference10_comment";
//   @file_put_contents($csv_file, $header . PHP_EOL, FILE_APPEND | LOCK_EX);
// }

$ok = @file_put_contents($csv_file, $line . PHP_EOL, FILE_APPEND | LOCK_EX);
if ($ok === false) {
  http_response_code(500);
  echo json_encode(array('ok' => false, 'error' => 'write_failed'));
  exit;
}

echo json_encode(array('ok' => true));


