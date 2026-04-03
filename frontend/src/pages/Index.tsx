import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  Download,
  FlaskConical,
  AlertCircle,
  Atom,
  BarChart3,
  Brain,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

// ── API Configuration ──────────────────────────────────────

const API_BASE_URL = "http://127.0.0.1:8000";

interface PredictionResult {
  toxicity: string;
  confidence: number;
  top_features: string[];
  molecule_image: string;
  report_url: string;
  ai_response: string;
}

const EXAMPLES = [
  { label: "Ethanol", smiles: "CCO" },
  { label: "Benzene", smiles: "c1ccccc1" },
  { label: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { label: "Acetaminophen", smiles: "CC(=O)NC1=CC=C(O)C=C1" },
  { label: "Caffeine", smiles: "Cn1cnc2c1c(=O)n(c(=O)n2C)C" },
];

const predictToxicity = async (smiles: string): Promise<PredictionResult> => {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(errorData?.detail || `Server error (${response.status})`);
  }

  return response.json();
};

// ── Component ──────────────────────────────────────────────

const Index = () => {
  const [smiles, setSmiles] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    if (!smiles.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await predictToxicity(smiles);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to get prediction. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    if (result?.report_url) {
      window.open(`${API_BASE_URL}${result.report_url}`, "_blank");
    }
  };

  const isToxic = result?.toxicity?.toLowerCase() === "toxic";

  return (
    <div className="min-h-screen bg-background grid-bg gradient-mesh">
      {/* Top Bar */}
      <header className="border-b border-border/50 bg-card/60 backdrop-blur-md sticky top-0 z-10">
        <div className="mx-auto max-w-6xl flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-lg bg-primary/15 flex items-center justify-center">
              <FlaskConical className="h-4.5 w-4.5 text-primary" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-foreground leading-none">
                CodeCure
              </h1>
              <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
                Drug Toxicity Predictor
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="text-[10px] border-primary/30 text-primary gap-1">
              <Sparkles className="h-3 w-3" /> Track A · AI Pharmacology
            </Badge>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {/* Hero Section */}
        <section className="mb-8 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground mb-2">
            Predict Drug Toxicity with <span className="text-primary">AI</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-sm leading-relaxed">
            Enter a SMILES string to analyze molecular toxicity using ML models.
            Get SHAP feature importance, molecule visualization, AI-powered analysis,
            and a downloadable PDF report.
          </p>
        </section>

        {/* Input */}
        <Card className="mb-6 glow-primary border-primary/20">
          <CardContent className="p-5">
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="flex-1">
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block uppercase tracking-wider">
                  SMILES Input
                </label>
                <Input
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  placeholder="e.g. CCO, c1ccccc1"
                  onKeyDown={(e) => e.key === "Enter" && handlePredict()}
                  className="font-mono text-sm bg-background border-border/60 h-11"
                />
              </div>
              <Button
                onClick={handlePredict}
                disabled={loading || !smiles.trim()}
                className="sm:self-end h-11 px-6 bg-primary text-primary-foreground hover:bg-primary/90 font-semibold"
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <>
                    <Brain className="h-4 w-4 mr-1.5" />
                    Predict
                  </>
                )}
              </Button>
            </div>
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Try:</span>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.smiles}
                  onClick={() => setSmiles(ex.smiles)}
                  className="text-xs px-2.5 py-1 rounded-md bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors font-mono"
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Error */}
        {error && (
          <Card className="mb-6 border-destructive/40 glow-danger">
            <CardContent className="flex items-center gap-3 p-4">
              <AlertCircle className="h-5 w-5 shrink-0 text-destructive" />
              <p className="text-sm text-destructive">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center gap-4 py-20">
            <div className="relative">
              <div className="h-16 w-16 rounded-full border-2 border-primary/20 flex items-center justify-center">
                <Atom className="h-8 w-8 text-primary animate-spin" style={{ animationDuration: "3s" }} />
              </div>
              <div className="absolute inset-0 h-16 w-16 rounded-full border-2 border-transparent border-t-primary animate-spin" />
            </div>
            <div className="text-center">
              <p className="text-sm text-foreground font-medium">Analyzing molecular structure…</p>
              <p className="text-xs text-muted-foreground mt-1">Computing descriptors, running model & generating AI analysis</p>
            </div>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="space-y-6 animate-in fade-in-0 slide-in-from-bottom-4 duration-500">

            {/* Prediction Card */}
            <Card className={`${isToxic ? "border-destructive/40 glow-danger" : "border-success/40 glow-success"}`}>
              <CardContent className="p-5 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`h-12 w-12 rounded-xl flex items-center justify-center ${isToxic ? "bg-destructive/15" : "bg-success/15"}`}>
                    {isToxic ? <ShieldAlert className="h-6 w-6 text-destructive" /> : <ShieldCheck className="h-6 w-6 text-success" />}
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider mb-0.5">Prediction</p>
                    <p className={`text-xl font-bold ${isToxic ? "text-destructive" : "text-success"}`}>{result.toxicity}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider mb-0.5">Confidence</p>
                  <p className="text-2xl font-bold text-foreground font-mono">{(result.confidence * 100).toFixed(1)}%</p>
                </div>
              </CardContent>
            </Card>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Molecule Image */}
              {result.molecule_image && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Atom className="h-4 w-4 text-accent" />
                      Molecule Structure
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex items-center justify-center p-4">
                    <div className="bg-white rounded-xl p-4 shadow-inner">
                      <img
                        src={`data:image/png;base64,${result.molecule_image}`}
                        alt="Molecule structure"
                        className="max-w-full h-auto max-h-[260px] object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Top Features (SHAP) */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-primary" />
                    Top Contributing Features (SHAP)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {result.top_features.map((feature, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <div className={`h-8 w-8 rounded-lg flex items-center justify-center text-xs font-bold ${
                          i === 0
                            ? "bg-primary/15 text-primary"
                            : "bg-secondary text-muted-foreground"
                        }`}>
                          #{i + 1}
                        </div>
                        <div className="flex-1">
                          <Badge
                            variant="outline"
                            className="font-mono text-xs border-primary/25 text-primary/90 bg-primary/5 px-3 py-1.5"
                          >
                            {feature}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-4 leading-relaxed">
                    These features were identified by SHAP (SHapley Additive exPlanations) as the most
                    influential molecular descriptors driving this prediction.
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* AI Analysis */}
            {result.ai_response && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Brain className="h-4 w-4 text-accent" />
                    AI-Powered Toxicity Analysis
                    <Badge variant="outline" className="text-[9px] ml-1 border-accent/30 text-accent">
                      Gemini
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-secondary/30 rounded-lg p-4 border border-border/40">
                    <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-line">
                      {result.ai_response}
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Download Report Button */}
            <Button
              variant="outline"
              className="w-full border-border/60 hover:border-primary/40 hover:bg-primary/5 h-12"
              onClick={handleDownloadReport}
            >
              <Download className="h-4 w-4 mr-2" />
              Download Full PDF Report
            </Button>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-16 pt-6 border-t border-border/30 text-center">
          <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
            CodeCure · AI Biohackathon · Track A — Drug Toxicity Prediction
          </p>
        </footer>
      </main>
    </div>
  );
};

export default Index;
